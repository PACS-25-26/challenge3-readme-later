# Scalability test: 1, 2, 4 MPI ranks, n=2^4..2^8
# Run from the project root: bash test/run_scalability.sh

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT_DIR}/test/data"

cd "${ROOT_DIR}"
make -j"$(nproc)"

for P in 1 2 4; do
    echo "=== P=${P} ==="
    mpirun --oversubscribe -np "${P}" ./benchmark \
        | tee test/data/results_p${P}.txt
    echo ""
done