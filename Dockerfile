# syntax=docker/dockerfile:1.7
FROM rust:1.85.1-bookworm AS builder

WORKDIR /workspace
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates clang cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY . .
ARG SOURCE_DATE_EPOCH=0
ENV CARGO_INCREMENTAL=0 \
    RUSTFLAGS="--remap-path-prefix=/workspace=."
RUN cargo build --locked --release --bin impossible-embedding \
    && mkdir -p /release/lib \
    && cp target/release/impossible-embedding /release/impossible-embedding \
    && find target/release -maxdepth 1 \( -type f -o -type l \) -name 'libonnxruntime.so*' -exec cp -P '{}' /release/lib/ \;

FROM debian:bookworm-slim AS runtime

ARG VERSION=0.0.0-dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="Impossible Embedding" \
      org.opencontainers.image.description="Local-first server for open embedding models" \
      org.opencontainers.image.source="https://github.com/impossibleG/impossible-embedding" \
      org.opencontainers.image.url="https://github.com/impossibleG/impossible-embedding" \
      org.opencontainers.image.documentation="https://github.com/impossibleG/impossible-embedding/blob/main/docs/operations.md" \
      org.opencontainers.image.licenses="MIT OR Apache-2.0" \
      org.opencontainers.image.version="$VERSION" \
      org.opencontainers.image.revision="$REVISION"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgcc-s1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 65532 impossible \
    && useradd --uid 65532 --gid 65532 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin impossible \
    && install -d -o impossible -g impossible /var/lib/impossible-embedding /etc/impossible-embedding

COPY --from=builder /release/impossible-embedding /usr/local/bin/impossible-embedding
COPY --from=builder /release/lib/ /usr/local/lib/
COPY --chown=root:root config/impossible-embedding.example.toml /etc/impossible-embedding/config.example.toml
COPY --chown=root:root LICENSE-MIT LICENSE-APACHE THIRD_PARTY_NOTICES.md THIRD_PARTY_LICENSES.txt /usr/share/licenses/impossible-embedding/
COPY --chown=root:root licenses/ONNXRUNTIME-LICENSE /usr/share/licenses/impossible-embedding/ONNXRUNTIME-LICENSE

ENV IMPOSSIBLE_CACHE_DIRECTORY=/var/lib/impossible-embedding \
    LD_LIBRARY_PATH=/usr/local/lib
WORKDIR /var/lib/impossible-embedding
VOLUME ["/var/lib/impossible-embedding"]
EXPOSE 8080 50051
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/impossible-embedding"]
CMD ["serve"]
