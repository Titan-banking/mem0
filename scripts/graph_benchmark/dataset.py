import json
import numpy as np
from pathlib import Path
import hashlib

def generate_dataset(size, tenant_count, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.Generator(np.random.PCG64(20261110))
    
    # Generate 64 normalized Gaussian 384d centers
    centers = rng.standard_normal((64, 384))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    
    # Entities
    entity_centers_idx = rng.integers(0, 64, size=size)
    entities_base = centers[entity_centers_idx]
    
    noise = rng.standard_normal((size, 384)) * 0.05
    entities = entities_base + noise
    entities = entities / np.linalg.norm(entities, axis=1, keepdims=True)
    entities = entities.astype(np.float32)
    
    tenants = np.arange(size) % tenant_count
    
    # Save entities
    np.save(output_dir / f"entities_{size}_{tenant_count}.npy", entities)
    
    # Edges: four distinct non-self outgoing neighbors within its tenant
    # Group entities by tenant
    tenant_to_entities = {t: [] for t in range(tenant_count)}
    for i in range(size):
        tenant_to_entities[tenants[i]].append(i)
        
    edges = []
    edge_id = 0
    for i in range(size):
        t = tenants[i]
        t_ents = tenant_to_entities[t]
        if len(t_ents) <= 4:
            neighbors = [n for n in t_ents if n != i]
        else:
            candidates = [n for n in t_ents if n != i]
            neighbors = rng.choice(candidates, size=4, replace=False)
            
        for n in neighbors:
            edges.append({
                "id": str(edge_id),
                "source": str(i),
                "destination": str(n),
                "tenant": str(t),
                "memory_id": f"m_{i}",
                "predicate": "related_to"
            })
            edge_id += 1
            
    with open(output_dir / f"edges_{size}_{tenant_count}.json", "w") as f:
        json.dump(edges, f)
        
    # Queries: 1,100 queries
    # Select 1,100 entities
    query_ent_idx = rng.choice(size, size=1100, replace=True)
    query_bases = entities[query_ent_idx]
    query_noise = rng.standard_normal((1100, 384)) * 0.01
    queries = query_bases + query_noise
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries.astype(np.float32)
    
    query_tenants = tenants[query_ent_idx]
    
    np.save(output_dir / f"queries_{size}_{tenant_count}.npy", queries)
    np.save(output_dir / f"query_tenants_{size}_{tenant_count}.npy", query_tenants)
    
    # Compute hashes
    h = hashlib.sha256()
    h.update(entities.tobytes())
    h.update(json.dumps(edges, sort_keys=True).encode())
    h.update(queries.tobytes())
    h.update(query_tenants.tobytes())
    
    return h.hexdigest()

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1]
    
    for size in [1000, 10000, 100000]:
        if size == 1000:
            tenants = [10]
        else:
            tenants = [10, 100]
            
        for t in tenants:
            hash_val = generate_dataset(size, t, output_dir)
            print(f"Generated {size} size, {t} tenants: {hash_val}")
