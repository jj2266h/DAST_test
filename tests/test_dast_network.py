import unittest

import torch

import DAST_test
from DAST_Network import DAST, DecoderLayer


class TestDecoderCausalMask(unittest.TestCase):
    def test_future_decoder_input_does_not_change_earlier_outputs(self):
        torch.manual_seed(7)
        layer = DecoderLayer(dim_val=4, dim_attn=4, n_heads=1, dropout=0.0)
        layer.eval()

        decoder_input = torch.randn(1, 3, 4)
        changed_input = decoder_input.clone()
        changed_input[:, -1, :] += 10.0
        encoder_output = torch.randn(1, 5, 4)

        original = layer(decoder_input, encoder_output)
        changed = layer(changed_input, encoder_output)

        torch.testing.assert_close(original[:, :-1], changed[:, :-1])

    def test_dropout_applies_to_every_decoder_residual_branch(self):
        torch.manual_seed(11)
        layer = DecoderLayer(dim_val=4, dim_attn=4, n_heads=1, dropout=1.0)
        layer.train()
        decoder_input = torch.randn(1, 3, 4)
        encoder_output = torch.randn(1, 5, 4)

        actual = layer(decoder_input, encoder_output)
        expected = layer.norm3(layer.norm2(layer.norm1(decoder_input)))

        torch.testing.assert_close(actual, expected)


class TestPaperOutputHead(unittest.TestCase):
    def _build_model(self, n_decoder_layers=1):
        return DAST(
            dim_val_s=64,
            dim_attn_s=64,
            dim_val_t=64,
            dim_attn_t=64,
            dim_val=64,
            dim_attn=64,
            time_step=42,
            input_size=14,
            dec_seq_len=10,
            out_seq_len=1,
            n_decoder_layers=n_decoder_layers,
            n_encoder_layers=2,
            n_heads=4,
            dropout=0.1,
        )

    def test_output_head_uses_64_unit_hidden_layer(self):
        model = self._build_model()

        self.assertTrue(hasattr(model, "output_hidden_fc"))
        self.assertEqual(model.output_hidden_fc.in_features, 10 * 64)
        self.assertEqual(model.output_hidden_fc.out_features, 64)
        self.assertEqual(model.out_fc.in_features, 64)
        self.assertEqual(model.out_fc.out_features, 1)

    def test_output_hidden_activation_is_relu(self):
        model = self._build_model()

        self.assertTrue(hasattr(model, "output_activation"))
        self.assertIsInstance(model.output_activation, torch.nn.ReLU)
        activated = model.output_activation(torch.tensor([-2.0, 0.0, 3.0]))
        torch.testing.assert_close(activated, torch.tensor([0.0, 0.0, 3.0]))

    def test_output_shape_is_preserved(self):
        model = self._build_model()

        output = model(torch.randn(2, 42, 14))
        self.assertEqual(tuple(output.shape), (2, 1))

    def test_default_dropout_matches_paper(self):
        model = DAST(
            64, 64, 64, 64, 64, 64,
            42, 14, 10, 1, 1, 2, 4,
        )

        self.assertEqual(model.sensor_encoder[0].dropout.p, 0.2)
        self.assertEqual(model.time_encoder[0].dropout.p, 0.2)
        self.assertEqual(model.decoder[0].dropout.p, 0.2)

    def test_all_configured_decoder_layers_run(self):
        model = self._build_model(n_decoder_layers=2)
        calls = []
        handles = [
            layer.register_forward_hook(
                lambda _module, _inputs, _output, index=index: calls.append(index)
            )
            for index, layer in enumerate(model.decoder)
        ]

        try:
            model(torch.randn(2, 42, 14))
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(calls, [0, 1])


class TestPredictionShape(unittest.TestCase):
    def test_singleton_batch_dimension_is_preserved(self):
        self.assertTrue(hasattr(DAST_test, "squeeze_last_dim"))

        squeezed = DAST_test.squeeze_last_dim(torch.randn(1, 1))

        self.assertEqual(tuple(squeezed.shape), (1,))


if __name__ == "__main__":
    unittest.main()
