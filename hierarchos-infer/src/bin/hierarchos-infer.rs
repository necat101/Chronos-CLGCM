use std::env;
use std::process::ExitCode;

use hierarchos_infer::Hierarchos;
use serde_json::json;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let Some(model_path) = args.next() else {
        return Err("usage: hierarchos-infer <model.hrf32> <token-ids> [--full-logits]".into());
    };
    let Some(token_arg) = args.next() else {
        return Err("missing comma-separated token IDs".into());
    };
    let full_logits = args.any(|arg| arg == "--full-logits");
    let tokens = token_arg
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if tokens.is_empty() {
        return Err("at least one token ID is required".into());
    }

    let model = Hierarchos::load(model_path)?;
    let mut state = model.new_state();
    let result = model.prefill(&mut state, &tokens)?;
    let (argmax, max_logit) = result
        .logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("model vocabulary is non-empty");
    let payload = if full_logits {
        json!({
            "argmax": argmax,
            "max_logit": max_logit,
            "logits": result.logits,
            "global_pos": state.global_pos,
        })
    } else {
        json!({
            "argmax": argmax,
            "max_logit": max_logit,
            "global_pos": state.global_pos,
        })
    };
    println!("{}", serde_json::to_string(&payload)?);
    Ok(())
}
