//! Reproducible, opt-in qualification for the first curated production model.
//!
//! The test is ignored because it downloads the immutable public model. Generate the independent
//! oracle file with `scripts/qualify_bge_oracle.py`, then set `IMPOSSIBLE_BGE_ORACLE_EVIDENCE` and
//! `IMPOSSIBLE_BGE_CACHE_DIR` before running this test with `--ignored --exact`.

use std::{borrow::Cow, env, fs, path::PathBuf};

use anyhow::{Context, Result, bail, ensure};
use impossible_embedding_core::{
    CancellationToken, EmbeddingBatch, ExecutionControl, RequestedModel,
};
use impossible_embedding_onnx::{EmbedOptions, EmbeddingTask, OnnxEmbeddingEngine, Truncation};
use impossible_models::{
    CancelToken, InstallOptions, Installer, ModelStatus, ModelStore, curated_manifests,
};
use serde_json::Value;
use tokenizers::Tokenizer;

const MODEL_ID: &str = "BAAI/bge-small-en-v1.5";
const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

fn required_path(name: &str) -> Result<PathBuf> {
    env::var_os(name)
        .map(PathBuf::from)
        .with_context(|| format!("{name} must name an explicit qualification path"))
}

fn cases(evidence: &Value) -> Result<&Vec<Value>> {
    evidence["cases"]
        .as_array()
        .context("oracle cases must be an array")
}

fn case_text(case: &Value) -> Result<&str> {
    case["text"].as_str().context("case text")
}

fn case_task(case: &Value) -> Result<EmbeddingTask> {
    match case["task"].as_str() {
        Some("document") => Ok(EmbeddingTask::Document),
        Some("query") => Ok(EmbeddingTask::Query),
        _ => bail!("unknown oracle task"),
    }
}

fn oracle_vector(case: &Value) -> Result<Vec<f32>> {
    case["embedding"]
        .as_array()
        .context("oracle embedding")?
        .iter()
        .map(|value| {
            value
                .to_string()
                .parse::<f32>()
                .ok()
                .context("oracle embedding element")
        })
        .collect()
}

fn embed(
    engine: &OnnxEmbeddingEngine,
    texts: &[&str],
    task: EmbeddingTask,
    truncation: Truncation,
) -> Result<Vec<Vec<f32>>> {
    let requested = RequestedModel::new(MODEL_ID)?;
    let batch = EmbeddingBatch::new(texts.iter().map(|text| Cow::Borrowed(*text)))?;
    let control = ExecutionControl::new(CancellationToken::default(), None);
    Ok(engine
        .embed_with_options(
            &requested,
            &batch,
            &control,
            EmbedOptions {
                task,
                truncation,
                ..EmbedOptions::default()
            },
        )?
        .vectors)
}

fn compare_vector(actual: &[f32], expected: &[f32], label: &str) -> Result<(f32, f32)> {
    ensure!(actual.len() == 384, "{label}: runtime width changed");
    ensure!(
        expected.len() == actual.len(),
        "{label}: oracle width changed"
    );
    let dot = actual
        .iter()
        .zip(expected)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = actual.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = expected
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let cosine = dot / (left_norm * right_norm);
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    ensure!(cosine >= 0.999_99, "{label}: cosine {cosine}");
    ensure!(max_abs <= 0.000_1, "{label}: maximum delta {max_abs}");
    Ok((cosine, max_abs))
}

fn verify_tokenizer(tokenizer: &Tokenizer, evidence: &Value) -> Result<()> {
    for case in cases(evidence)? {
        let text = case_text(case)?;
        let prepared = if case_task(case)? == EmbeddingTask::Query {
            Cow::Owned(format!("{QUERY_PREFIX}{text}"))
        } else {
            Cow::Borrowed(text)
        };
        let actual = tokenizer
            .encode(prepared.as_ref(), true)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let expected = case["token_ids"]
            .as_array()
            .context("oracle token ids")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|id| u32::try_from(id).ok())
                    .context("oracle token id")
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(actual.get_ids() == expected, "tokenizer ids changed");
    }
    Ok(())
}

fn record_metrics(min_cosine: &mut f32, max_abs: &mut f32, metrics: (f32, f32)) {
    *min_cosine = min_cosine.min(metrics.0);
    *max_abs = max_abs.max(metrics.1);
}

fn verify_embeddings(engine: &OnnxEmbeddingEngine, evidence: &Value) -> Result<(f32, f32)> {
    let mut min_cosine = 1.0_f32;
    let mut max_abs = 0.0_f32;
    for task in [EmbeddingTask::Document, EmbeddingTask::Query] {
        let selected = cases(evidence)?
            .iter()
            .filter(|case| case_task(case).ok() == Some(task))
            .collect::<Vec<_>>();
        let texts = selected
            .iter()
            .map(|case| case_text(case))
            .collect::<Result<Vec<_>>>()?;
        let batched = embed(engine, &texts, task, Truncation::Reject)?;
        for (case, batch_vector) in selected.iter().zip(&batched) {
            let text = case_text(case)?;
            let single = embed(engine, &[text], task, Truncation::Reject)?;
            let name = case["name"].as_str().context("case name")?;
            record_metrics(
                &mut min_cosine,
                &mut max_abs,
                compare_vector(batch_vector, &oracle_vector(case)?, name)?,
            );
            record_metrics(
                &mut min_cosine,
                &mut max_abs,
                compare_vector(&single[0], batch_vector, &format!("{name} batch/single"))?,
            );
        }
    }

    let query = "What is the capital of France?";
    let raw = embed(engine, &[query], EmbeddingTask::Query, Truncation::Reject)?;
    let prefixed = embed(
        engine,
        &[&format!("{QUERY_PREFIX}{query}")],
        EmbeddingTask::Query,
        Truncation::Reject,
    )?;
    record_metrics(
        &mut min_cosine,
        &mut max_abs,
        compare_vector(&raw[0], &prefixed[0], "query prefix exactly once")?,
    );

    let exact = std::iter::repeat_n("hello", 510)
        .collect::<Vec<_>>()
        .join(" ");
    let over = std::iter::repeat_n("hello", 511)
        .collect::<Vec<_>>()
        .join(" ");
    let exact_vector = embed(
        engine,
        &[&exact],
        EmbeddingTask::Document,
        Truncation::Reject,
    )?;
    ensure!(
        embed(
            engine,
            &[&over],
            EmbeddingTask::Document,
            Truncation::Reject
        )
        .is_err(),
        "513 tokens must be rejected"
    );
    let truncated = embed(
        engine,
        &[&over],
        EmbeddingTask::Document,
        Truncation::Truncate,
    )?;
    record_metrics(
        &mut min_cosine,
        &mut max_abs,
        compare_vector(
            &truncated[0],
            &exact_vector[0],
            "special-token-preserving truncation",
        )?,
    );
    Ok((min_cosine, max_abs))
}

#[tokio::test]
#[ignore = "downloads the pinned public model and requires an independently generated oracle"]
async fn pinned_bge_install_verify_embed_and_offline_restart() -> Result<()> {
    let cache = required_path("IMPOSSIBLE_BGE_CACHE_DIR")?;
    let evidence_path = required_path("IMPOSSIBLE_BGE_ORACLE_EVIDENCE")?;
    let evidence: Value = serde_json::from_slice(&fs::read(evidence_path)?)?;
    ensure!(evidence["model"] == MODEL_ID, "oracle model changed");
    ensure!(
        evidence["revision"] == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "oracle revision changed"
    );

    let manifest = curated_manifests()?
        .into_iter()
        .find(|candidate| candidate.canonical_id == MODEL_ID)
        .context("curated BGE manifest")?;
    ensure!(manifest.semantics_verified(), "manifest is not qualified");
    for artifact in &manifest.artifacts {
        let oracle = &evidence["artifacts"][&artifact.path];
        ensure!(
            oracle["sha256"] == artifact.sha256 && oracle["size"] == artifact.size,
            "oracle artifact identity changed"
        );
    }

    let store = ModelStore::new(&cache)?;
    let installer = Installer::new(store.clone(), InstallOptions::default())?;
    ensure!(
        installer.install(&manifest, &CancelToken::new()).await? == ModelStatus::Loadable,
        "online install did not become loadable"
    );
    let verified = store.verified_model(&manifest)?;
    let tokenizer = Tokenizer::from_file(verified.artifact_path("tokenizer.json")?)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    verify_tokenizer(&tokenizer, &evidence)?;
    let engine = OnnxEmbeddingEngine::new();
    engine.load(&verified)?;
    engine.warm()?;
    let (min_cosine, max_abs) = verify_embeddings(&engine, &evidence)?;
    println!("qualification min_cosine={min_cosine:.9} max_abs={max_abs:.9}");
    engine.unload()?;
    drop(verified);
    drop(store);

    let restarted_store = ModelStore::new(&cache)?;
    let offline = Installer::new(
        restarted_store.clone(),
        InstallOptions {
            offline: true,
            ..InstallOptions::default()
        },
    )?;
    ensure!(
        offline.install(&manifest, &CancelToken::new()).await? == ModelStatus::Loadable,
        "offline restart did not reuse the exact verified identity"
    );
    let restarted_verified = restarted_store.verified_model(&manifest)?;
    let restarted = OnnxEmbeddingEngine::new();
    restarted.load(&restarted_verified)?;
    let first = cases(&evidence)?.first().context("first oracle case")?;
    let output = embed(
        &restarted,
        &[case_text(first)?],
        case_task(first)?,
        Truncation::Reject,
    )?;
    compare_vector(&output[0], &oracle_vector(first)?, "offline restart")?;
    Ok(())
}
