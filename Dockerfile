FROM rust:latest

RUN apt-get update && apt-get install -y \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN cargo build --release

CMD ["./target/release/rinex_timer", \
    "./rinex/v2/obs/cgtc0920.14o", \
    "./rinex/v3/obs/ASIR00ITA_R_20242810000_01D_30S_MO.crx"]