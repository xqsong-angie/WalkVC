#!/usr/bin/env bash
# Load a backup made by scripts/backup.sh. OVERWRITES the local Mongo db and the Neo4j 'neo4j' database.
#   scripts/restore.sh [in_dir]      (default: backups/)
# For production first: export COMPOSE_FILE=docker-compose.yml:docker-compose.prod.yml
set -euo pipefail
source "$(dirname "$0")/_dbutil.sh"

IN="${1:-backups}"
IN_ABS="$(cd "$IN" && pwd)"

echo ">> verifying checksums"
(cd "$IN" && sha256 -c SHA256SUMS)

echo ">> mongorestore $MONGO_DB (drops existing collections first)"
docker compose up -d mongo
wait_mongo
docker compose exec -T mongo mongorestore ${MONGO_AUTH[@]+"${MONGO_AUTH[@]}"} --drop --archive --gzip < "$IN/vcgraph.mongo.gz"

echo ">> neo4j load"
docker compose stop neo4j 2>/dev/null || true
docker compose run --rm --no-deps -v "$IN_ABS":/backups neo4j \
  neo4j-admin database load neo4j --from-path=/backups --overwrite-destination=true

docker compose up -d
wait_neo4j

echo ">> counts here (must match $IN/counts.txt)"
print_counts | tee "$IN/counts.restored.txt"
if diff -q "$IN/counts.txt" "$IN/counts.restored.txt" >/dev/null; then
  echo "OK: counts match the source machine"
else
  echo "WARNING: counts differ:"; diff "$IN/counts.txt" "$IN/counts.restored.txt" || true
fi
