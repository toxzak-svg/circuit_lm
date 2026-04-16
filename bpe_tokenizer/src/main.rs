//! Fast BPE Tokenizer — Streaming Rust implementation
//! 
//! Key design: counts pairs from FILE (streaming), never stores full text in memory.
//!
//! Usage:
//!   bpe_tokenizer build <text_file> <vocab_size> <output_json>
//!   bpe_tokenizer encode <vocab_json> <text_file>
//!   bpe_tokenizer decode <vocab_json> <token_ids_csv>

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use anyhow::{Context, Result};

const PAD_TOKEN: &str = "<PAD>";
const UNK_TOKEN: &str = "<UNK>";
const CHUNK_SIZE: usize = 1024 * 1024; // 1MB

// ---------------------------------------------------------------------------
// Tokenizer (serde-compatible)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Tokenizer {
    vocab: Vec<String>,
    max_piece_len: usize,
}

impl Tokenizer {
    fn new() -> Self {
        let vocab = vec![PAD_TOKEN.to_string(), UNK_TOKEN.to_string()];
        Tokenizer { vocab, max_piece_len: 1 }
    }

    fn from_dict(data: &serde_json::Value) -> Result<Self> {
        let pieces = data["pieces"].as_array().context("Missing 'pieces'")?;
        let mut vocab: Vec<String> = pieces.iter()
            .map(|p| p.as_str().unwrap_or("<PAD>").to_string())
            .collect();
        vocab.insert(0, PAD_TOKEN.to_string());
        vocab.insert(1, UNK_TOKEN.to_string());
        let max_piece_len = vocab.iter().map(|s| s.len()).max().unwrap_or(1);
        Ok(Tokenizer { vocab, max_piece_len })
    }

    fn from_json_file(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let data: serde_json::Value = serde_json::from_str(&text)?;
        Self::from_dict(&data)
    }

    fn to_dict(&self) -> serde_json::Value {
        serde_json::json!({
            "mode": "bpe",
            "pieces": &self.vocab[2..],
        })
    }

    fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.to_dict())?;
        fs::write(path, json)?;
        Ok(())
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::with_capacity(text.len() / 2);
        let bytes = text.as_bytes();
        let n = bytes.len();
        let mut i = 0;

        while i < n {
            let remaining = n - i;
            let max_len = self.max_piece_len.min(remaining);
            let mut best_id: Option<usize> = None;
            let mut best_len = 0;

            for len in (1..=max_len).rev() {
                let piece = &bytes[i..i+len];
                let piece_str = String::from_utf8_lossy(piece).into_owned();
                if let Some(pos) = self.vocab.iter().position(|v| v == &piece_str) {
                    best_id = Some(pos);
                    best_len = len;
                    break;
                }
            }

            match best_id {
                Some(id) => { ids.push(id); i += best_len; }
                None => { ids.push(1); i += 1; }
            }
        }
        ids
    }

    fn decode(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        for &id in ids {
            if id < self.vocab.len() {
                let piece = &self.vocab[id];
                if piece != PAD_TOKEN && piece != UNK_TOKEN {
                    out.push_str(piece);
                }
            }
        }
        out
    }

    fn vocab_size(&self) -> usize { self.vocab.len() }
}

// ---------------------------------------------------------------------------
// Streaming BPE builder
// ---------------------------------------------------------------------------

fn count_pairs_in_text(text: &str, vocab_set: &HashMap<String, bool>) -> HashMap<(String,String), usize> {
    let mut pair_counts = HashMap::new();
    let chars: Vec<char> = text.chars().collect();
    for window in chars.windows(2) {
        let a = window[0].to_string();
        let b = window[1].to_string();
        if vocab_set.contains_key(&a) && vocab_set.contains_key(&b) {
            *pair_counts.entry((a, b)).or_insert(0) += 1;
        }
    }
    pair_counts
}

fn apply_merge_in_text(text: &str, pair: &(String, String)) -> String {
    let merged = format!("{}{}", pair.0, pair.1);
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::with_capacity(chars.len() * 2);
    let mut i = 0;
    while i < chars.len() {
        if i + 1 < chars.len() {
            let a = chars[i].to_string();
            let b = chars[i+1].to_string();
            if a == pair.0 && b == pair.1 {
                result.push_str(&merged);
                i += 2;
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

fn build_bpe_vocab_streaming(text_path: &str, target_vocab_size: usize) -> Result<Tokenizer> {
    println!("Collecting character frequencies (streaming)...");
    let mut char_freq: HashMap<char, usize> = HashMap::new();

    {
        let file = File::open(text_path)?;
        let reader = BufReader::with_capacity(CHUNK_SIZE, file);
        let mut line = String::new();
        for result in reader.lines() {
            line.clear();
            if let Ok(l) = result { line.push('\n'); line.push_str(&l); }
            for ch in line.chars() {
                *char_freq.entry(ch).or_insert(0) += 1;
            }
        }
    }
    println!("  {} unique characters", char_freq.len());

    // Build base vocab from character frequencies
    let mut char_counts: Vec<(char, usize)> = char_freq.into_iter().collect();
    char_counts.sort_by(|a, b| b.1.cmp(&a.1));
    let max_chars = target_vocab_size.saturating_sub(2);

    let mut vocab: Vec<String> = vec![PAD_TOKEN.to_string(), UNK_TOKEN.to_string()];
    let mut vocab_set: HashMap<String, bool> = HashMap::new();
    vocab_set.insert(PAD_TOKEN.to_string(), true);
    vocab_set.insert(UNK_TOKEN.to_string(), true);

    for (ch, _) in char_counts.iter().take(max_chars) {
        let s = ch.to_string();
        vocab_set.insert(s.clone(), true);
        vocab.push(s);
    }
    println!("  {} chars in base vocab", vocab.len());

    let mut merges_needed = target_vocab_size.saturating_sub(vocab.len());
    println!("  {} merges to perform", merges_needed);

    // Working text stored as lines
    let tmp_in = format!("{}.bpe.tmp", text_path);
    let tmp_out = format!("{}.bpe.tmp2", text_path);

    // Copy original to tmp_in
    fs::copy(text_path, &tmp_in)?;

    let mut round = 0;
    while merges_needed > 0 {
        round += 1;
        if round % 20 == 0 || merges_needed < 20 {
            println!("  round {:4}, {} tokens, {} merges left...", round, vocab.len(), merges_needed);
        }

        // Stream through tmp_in counting pairs
        let file = File::open(&tmp_in)?;
        let reader = BufReader::with_capacity(CHUNK_SIZE, file);
        let mut line = String::new();
        let mut pair_counts: HashMap<(String,String), usize> = HashMap::new();

        for result in reader.lines() {
            line.clear();
            if let Ok(l) = result { line = l; }
            let pairs = count_pairs_in_text(&line, &vocab_set);
            for (k, v) in pairs {
                *pair_counts.entry(k).or_insert(0) += v;
            }
        }

        // Find best valid pair
        let mut pairs: Vec<_> = pair_counts.iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(a.1));

        let mut best_pair: Option<(String, String)> = None;
        for ((a, b), &count) in pairs {
            if count < 2 { break; }
            let merged = format!("{}{}", a, b);
            if !vocab_set.contains_key(&merged) {
                best_pair = Some((a.clone(), b.clone()));
                break;
            }
        }

        let Some(pair) = best_pair else {
            println!("  No more valid pairs. Stopping at {} tokens.", vocab.len());
            break;
        };

        let merged_str = format!("{}{}", pair.0, pair.1);
        vocab_set.insert(merged_str.clone(), true);
        vocab.push(merged_str.clone());
        merges_needed -= 1;

        // Apply merge to tmp_in → tmp_out (streaming)
        let f_in = File::open(&tmp_in)?;
        let reader = BufReader::with_capacity(CHUNK_SIZE, f_in);
        let f_out = File::create(&tmp_out)?;
        let mut wtr = std::io::BufWriter::new(f_out);
        let mut line = String::new();

        for result in reader.lines() {
            line.clear();
            if let Ok(l) = result { line = l; }
            let merged_line = apply_merge_in_text(&line, &pair);
            writeln!(wtr, "{}", merged_line)?;
        }
        wtr.flush()?;
        fs::rename(&tmp_out, &tmp_in)?; // overwrite tmp_in with merged version
    }

    let _ = fs::remove_file(&tmp_out);
    let _ = fs::remove_file(&tmp_in);

    let mut tokenizer = Tokenizer::new();
    tokenizer.vocab = vocab;
    tokenizer.max_piece_len = tokenizer.vocab.iter().map(|s| s.len()).max().unwrap_or(1);

    Ok(tokenizer)
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn cmd_build(text_path: &str, vocab_size: usize, output_path: &str) -> Result<()> {
    println!("=== Fast BPE Builder (Streaming Rust) ===");
    let t0 = std::time::Instant::now();
    let tokenizer = build_bpe_vocab_streaming(text_path, vocab_size)?;
    println!("Built {} tokens in {:.1}s", tokenizer.vocab_size(), t0.elapsed().as_secs_f64());
    tokenizer.save(Path::new(output_path))?;
    println!("Saved to {}", output_path);
    Ok(())
}

fn cmd_encode(vocab_path: &str, text_path: &str) -> Result<()> {
    let tokenizer = Tokenizer::from_json_file(Path::new(vocab_path))?;
    let text = fs::read_to_string(text_path)?;
    let ids = tokenizer.encode(&text);
    println!("Encoded {} chars → {} tokens", text.len(), ids.len());
    println!("First 20: {:?}", &ids[..ids.len().min(20)]);
    Ok(())
}

fn cmd_decode(vocab_path: &str, ids_csv: &str) -> Result<()> {
    let tokenizer = Tokenizer::from_json_file(Path::new(vocab_path))?;
    let ids: Vec<usize> = ids_csv
        .split(|c| c == ',' || c == ' ' || c == '\n')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().parse().unwrap_or(1))
        .collect();
    let text = tokenizer.decode(&ids);
    println!("Decoded {} tokens → {} chars", ids.len(), text.len());
    println!("Text: {}", &text[..text.len().min(200)]);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Fast BPE Tokenizer (Rust - streaming)");
        eprintln!("Usage:");
        eprintln!("  bpe_tokenizer build <text_file> <vocab_size> <output_json>");
        eprintln!("  bpe_tokenizer encode <vocab_json> <text_file>");
        eprintln!("  bpe_tokenizer decode <vocab_json> <token_ids_csv>");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "build" => {
            if args.len() != 5 { eprintln!("Usage: build <text> <vocab_size> <output>"); std::process::exit(1); }
            let vocab_size: usize = args[3].parse().context("vocab_size must be a number")?;
            cmd_build(&args[2], vocab_size, &args[4])?;
        }
        "encode" => {
            if args.len() != 4 { eprintln!("Usage: encode <vocab> <text>"); std::process::exit(1); }
            cmd_encode(&args[2], &args[3])?;
        }
        "decode" => {
            if args.len() != 4 { eprintln!("Usage: decode <vocab> <ids>"); std::process::exit(1); }
            cmd_decode(&args[2], &args[3])?;
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }
    Ok(())
}
