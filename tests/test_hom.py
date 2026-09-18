import numpy as np
import perceval as pcvl


def test_hong_ou_mandel_limit():
    circuit = pcvl.Circuit(2).add((0, 1), pcvl.BS())
    processor = pcvl.Processor("SLOS", circuit)
    processor.with_input(pcvl.BasicState([1, 1]))
    result = pcvl.algorithm.Sampler(processor).probs()["results"]
    probabilities = {str(state): float(value) for state, value in result.items()}
    assert probabilities.get("|1,1>", 0) < 1e-10
    assert np.isclose(probabilities["|2,0>"], 0.5)
    assert np.isclose(probabilities["|0,2>"], 0.5)
