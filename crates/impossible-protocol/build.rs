//! Reproducible protobuf generation using a workspace-managed compiler.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    let mut prost = tonic_prost_build::Config::new();
    prost.protoc_executable(protoc);

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .build_transport(false)
        .compile_with_config(prost, &["proto/embedding.proto"], &["proto"])?;

    println!("cargo:rerun-if-changed=proto/embedding.proto");
    Ok(())
}
