import numpy as np

from core.model_backends.onnx_backend import ONNXBackend


class _FakeMeta:
    def __init__(self, name, shape, type_str):
        self.name = name
        self.shape = shape
        self.type = type_str


class _FakeSession:
    def __init__(self):
        self._inputs = [
            _FakeMeta("tokens", [1, 1], "tensor(int64)"),
            _FakeMeta("state_in", [1], "tensor(float)"),
        ]
        self._outputs = [
            _FakeMeta("logits", [1, 1, 256], "tensor(float)"),
            _FakeMeta("state_out", [1], "tensor(float)"),
        ]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, _, feeds):
        tok_arr = feeds["tokens"]
        tok = int(np.asarray(tok_arr).reshape(-1)[-1])
        logits = np.zeros((1, 1, 256), dtype=np.float32)
        logits[0, 0, (tok + 1) % 256] = 10.0
        state = np.array([tok], dtype=np.float32)
        return [logits, state]


def test_onnx_backend_classifies_fake_graph():
    b = ONNXBackend()
    b.session = _FakeSession()
    b.input_names = [i.name for i in b.session.get_inputs()]
    b.output_names = [o.name for o in b.session.get_outputs()]
    b._classify_graph()
    assert b.graph_ready is True
    assert b.token_input_name == "tokens"
    assert b.logits_output_name == "logits"


def test_onnx_backend_generates_with_fake_graph():
    b = ONNXBackend()
    b.session = _FakeSession()
    b.input_names = [i.name for i in b.session.get_inputs()]
    b.output_names = [o.name for o in b.session.get_outputs()]
    b._classify_graph()
    text, tokens, elapsed = b.generate("A", max_tokens=3, temperature=0.00001, top_p=1.0, top_k=0)
    assert isinstance(text, str)
    assert tokens == 3
    assert elapsed > 0
