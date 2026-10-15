"""Copy authored meshes only to byte-identical refined TMD viewport peers.

The first OBJ comment names its EMD and offset; every remaining OBJ line must
match, including element layout, positions, normals, colours, UVs and faces.
The replacement format itself is keyed by the refined source hash. Candidate
copies go to a separate directory for verification and viewer inspection.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct


def mesh_hash(path):
    with path.open() as source:
        for _ in range(8):
            line = source.readline()
            if line.startswith("# re1_mesh_hash "):
                return int(line.split()[2], 16)
    raise ValueError(f"{path}: no mesh hash")


def canonical_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for line in source:
            if not line.startswith(b"# RE1 TMD viewport geometry: "):
                digest.update(line)
    return digest.digest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--replacements", type=Path,
                        default=Path("tools/data/entity_mesh_replacements"))
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    source = options.refined_dir.resolve()
    installed = options.replacements.resolve()
    output = options.out.resolve()
    output.mkdir(parents=True, exist_ok=True)

    by_hash = defaultdict(list)
    all_objs = {}
    for path in source.glob("*_refined.obj"):
        key = path.name.removesuffix("_refined.obj")
        if key in all_objs:
            raise ValueError(f"duplicate refined OBJ {key}")
        all_objs[key] = path
        by_hash[mesh_hash(path)].append(path)

    digest_cache = {}
    def digest(path):
        if path not in digest_cache:
            digest_cache[path] = canonical_digest(path)
        return digest_cache[path]

    proposals = {}
    for replacement in sorted(installed.glob("*.r1mesh")):
        key = replacement.stem
        origin = all_objs.get(key)
        if origin is None:
            raise ValueError(f"{replacement}: no refined OBJ")
        data = replacement.read_bytes()
        if len(data) < 16 or data[:8] != b"R1MESH01":
            raise ValueError(f"{replacement}: bad replacement header")
        source_hash = struct.unpack_from("<I", data, 8)[0]
        if source_hash != mesh_hash(origin):
            raise ValueError(f"{replacement}: source hash mismatch")
        expected = digest(origin)
        for peer in by_hash[source_hash]:
            peer_key = peer.name.removesuffix("_refined.obj")
            destination_name = f"{peer_key}.r1mesh"
            if peer_key == key or (installed / destination_name).exists():
                continue
            actual = digest(peer)
            if actual != expected:
                continue
            prior = proposals.get(destination_name)
            if prior is not None and prior[0] != data:
                raise ValueError(f"different authored meshes target {destination_name}")
            proposals[destination_name] = (data, replacement.name, peer.name)

    manifest = []
    for name, (data, source_name, peer_name) in sorted(proposals.items()):
        path = output / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"existing candidate differs: {path}")
        path.write_bytes(data)
        manifest.append({"file": name, "copied_from": source_name,
                         "matching_refined_obj": peer_name})
        print(f"{source_name} -> {name}")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(manifest)} identical-geometry candidates")


if __name__ == "__main__":
    main()
