import numpy as np
from flwr.common import ndarrays_to_parameters

from project.fed.server.strategy.fedavgNZ import aggregate
from project.fed.transport.sparse_codec import decode_parameters, encode_parameters, estimate_encoded_size

BASE_CFG = {
    "enabled": True,
    "tensor_type": "sparse_transport_v1",
    "csr_min_sparsity": 0.85,
    "dense_fallback_if_not_smaller": True,
    "index_dtype": "int32",
    "bitorder": "little",
}

def test_dense_round_trip():
    arr = np.arange(12, dtype=np.float32).reshape(3,4)
    cfg = dict(BASE_CFG, task_sparsity=0.1)
    p = encode_parameters([arr], cfg)
    out = decode_parameters(p)[0]
    np.testing.assert_array_equal(arr, out)

def test_bitmap_roundtrip_1d_2d_4d():
    cfg = dict(BASE_CFG, task_sparsity=0.5)
    arrays = [np.array([0,1,0,2],dtype=np.float32), np.array([[0,1],[2,0]],dtype=np.float32), np.zeros((2,2,2,2),dtype=np.float32)]
    arrays[2][0,1,1,1]=3
    p = encode_parameters(arrays,cfg)
    outs = decode_parameters(p)
    for a,b in zip(arrays,outs): np.testing.assert_array_equal(a,b)

def test_csr_roundtrip_2d_and_4d():
    cfg = dict(BASE_CFG, task_sparsity=0.9)
    a2 = np.zeros((4,5),dtype=np.float32); a2[1,2]=1; a2[3,4]=2
    a4 = np.zeros((3,2,2,2),dtype=np.float32); a4[1,1,1,1]=3
    p = encode_parameters([a2,a4],cfg)
    o2,o4 = decode_parameters(p)
    np.testing.assert_array_equal(a2,o2)
    np.testing.assert_array_equal(a4,o4)

def test_fallback_dense_when_not_smaller():
    arr = np.ones((8,),dtype=np.float32)
    cfg = dict(BASE_CFG, task_sparsity=0.5)
    info = estimate_encoded_size([arr],cfg)
    assert info["encoding_counts"]["dense"] == 1

def test_auto_selection_modes():
    arr = np.zeros((4,4),dtype=np.float32); arr[0,0]=1
    info_csr = estimate_encoded_size([arr], dict(BASE_CFG, task_sparsity=0.9, dense_fallback_if_not_smaller=False))
    assert info_csr["encoding_counts"]["csr"] == 1
    info_bitmap = estimate_encoded_size([arr], dict(BASE_CFG, task_sparsity=0.5, dense_fallback_if_not_smaller=False))
    assert info_bitmap["encoding_counts"]["bitmap_values"] == 1

def test_packbits_size():
    arr = np.array([1,0,0,0,0,0,0,0,1],dtype=np.float32)
    p = encode_parameters([arr], dict(BASE_CFG, task_sparsity=0.5, dense_fallback_if_not_smaller=False))
    payload = p.tensors[0]
    hlen = int.from_bytes(payload[:4],"little")
    import json
    meta = json.loads(payload[4:4+hlen])
    assert meta["buffers"][0]["nbytes"] == 2

def test_aggregation_parity_dense_vs_encoded_decode():
    w1 = [np.array([1,0,3],dtype=np.float32)]
    w2 = [np.array([2,0,1],dtype=np.float32)]
    dense = aggregate([(w1,10),(w2,30)])
    cfg = dict(BASE_CFG, task_sparsity=0.5)
    e1 = decode_parameters(encode_parameters(w1,cfg))
    e2 = decode_parameters(encode_parameters(w2,cfg))
    sparse_flow = aggregate([(e1,10),(e2,30)])
    np.testing.assert_allclose(dense[0], sparse_flow[0], rtol=1e-6, atol=1e-6)
