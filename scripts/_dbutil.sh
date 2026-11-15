# Shared helpers for backup.sh / restore.sh (sourced, not run).
# Docker compose picks up COMPOSE_FILE from the environment, so the prod override applies here too.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# envval KEY DEFAULT : value from the environment, else from .env, else DEFAULT (no `source .env`:
# values such as SEC_USER_AGENT contain spaces)
envval() {
  local v="${!1:-}"
  if [ -z "$v" ] && [ -f .env ]; then
    v="$(grep -E "^$1=" .env | tail -1 | cut -d= -f2- | sed -e 's/^["'\'']//' -e 's/["'\'']$//')"
  fi
  echo "${v:-$2}"
}

MONGO_DB="$(envval MONGO_DB vcgraph)"
NEO4J_PASSWORD="$(envval NEO4J_PASSWORD password123)"
MONGO_ROOT_USER="$(envval MONGO_ROOT_USER "")"
MONGO_ROOT_PASSWORD="$(envval MONGO_ROOT_PASSWORD "")"
MONGO_AUTH=()
if [ -n "$MONGO_ROOT_USER" ]; then
  MONGO_AUTH=(--username "$MONGO_ROOT_USER" --password "$MONGO_ROOT_PASSWORD" --authenticationDatabase admin)
fi

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

wait_mongo() {
  for _ in $(seq 60); do
    docker compose exec -T mongo mongosh --quiet ${MONGO_AUTH[@]+"${MONGO_AUTH[@]}"} --eval 'db.runCommand({ping:1}).ok' >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "mongo did not come up" >&2; return 1
}

wait_neo4j() {
  for _ in $(seq 90); do
    docker compose exec -T neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "RETURN 1" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "neo4j did not come up (wrong NEO4J_PASSWORD?)" >&2; return 1
}

# Row counts used to check that a restore matches the source machine.
print_counts() {
  echo "-- mongo ($MONGO_DB)"
  docker compose exec -T mongo mongosh --quiet ${MONGO_AUTH[@]+"${MONGO_AUTH[@]}"} "$MONGO_DB" --eval \
    'db.getCollectionNames().sort().forEach(c => print(c + " " + db.getCollection(c).countDocuments()))'
  echo "-- neo4j"
  docker compose exec -T neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" --format plain \
    "MATCH (n:Entity) WITH [l IN labels(n) WHERE l <> 'Entity'][0] AS label, count(*) AS c RETURN label, c ORDER BY label"
  docker compose exec -T neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" --format plain \
    "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS c ORDER BY rel"
}
