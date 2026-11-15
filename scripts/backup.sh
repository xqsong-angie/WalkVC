#!/usr/bin/env bash
# Export the Mongo database and the Neo4j graph so another machine can run without re-crawling.
#   scripts/backup.sh [out_dir]      (default: backups/)
# Produces vcgraph.mongo.gz, neo4j.dump, counts.txt, SHA256SUMS. Neo4j is stopped for ~1 min
# (Community Edition can only dump an offline database) and started again afterwards.
set -euo pipefail
source "$(dirname "$0")/_dbutil.sh"

OUT="${1:-backups}"
mkdir -p "$OUT"
OUT_ABS="$(cd "$OUT" && pwd)"

docker compose up -d mongo neo4j
wait_mongo
wait_neo4j

echo ">> counts (compare these after restoring)"
print_counts | tee "$OUT/counts.txt"

echo ">> mongodump $MONGO_DB"
docker compose exec -T mongo mongodump ${MONGO_AUTH[@]+"${MONGO_AUTH[@]}"} --db "$MONGO_DB" --archive --gzip > "$OUT/vcgraph.mongo.gz"

echo ">> neo4j dump (stopping neo4j)"
docker compose stop neo4j
docker compose run --rm --no-deps -v "$OUT_ABS":/backups neo4j \
  neo4j-admin database dump neo4j --to-path=/backups --overwrite-destination=true
docker compose start neo4j

(cd "$OUT" && sha256 vcgraph.mongo.gz neo4j.dump > SHA256SUMS)
ls -lh "$OUT"
echo "done. Send the whole '$OUT' folder (not via git)."
