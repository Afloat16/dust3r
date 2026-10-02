import unittest

import torch

from dust3r.utils.geometry import normalize_pointcloud
from dust3r.utils.misc import invalid_to_zeros


class TestPointcloudNormalizationCounts(unittest.TestCase):
    def test_implicit_point_count_matches_all_valid_mask(self):
        for shape in ((2, 5, 3), (2, 3, 4, 3), (2, 2, 3, 4, 3)):
            with self.subTest(shape=shape):
                pts = torch.ones(shape, dtype=torch.float64)
                implicit, implicit_count = invalid_to_zeros(pts, None, ndim=3)
                explicit, explicit_count = invalid_to_zeros(
                    pts, torch.ones(shape[:-1], dtype=torch.bool), ndim=3
                )
                torch.testing.assert_close(implicit, explicit)
                self.assertEqual(implicit_count, explicit_count[0].item())

    def test_default_normalization_has_unit_mean_distance(self):
        pts = torch.tensor([[[[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]]]])
        actual = normalize_pointcloud(pts, None)
        torch.testing.assert_close(actual.norm(dim=-1).mean(), torch.tensor(1.0))

    def test_unmasked_and_all_valid_values_and_gradients_match(self):
        generator = torch.Generator().manual_seed(5)
        for dtype in (torch.float32, torch.float64):
            for mode in ("avg_dis", "avg_log1p"):
                with self.subTest(dtype=dtype, mode=mode):
                    pts1 = torch.randn(
                        2, 3, 4, 3, generator=generator, dtype=dtype
                    ).requires_grad_()
                    pts2 = torch.randn(
                        2, 1, 5, 3, generator=generator, dtype=dtype
                    ).requires_grad_()
                    actual = normalize_pointcloud(pts1, pts2, norm_mode=mode)
                    expected = normalize_pointcloud(
                        pts1,
                        pts2,
                        norm_mode=mode,
                        valid1=torch.ones(pts1.shape[:-1], dtype=torch.bool),
                        valid2=torch.ones(pts2.shape[:-1], dtype=torch.bool),
                    )
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b)
                    upstream = [
                        torch.randn(t.shape, generator=generator, dtype=dtype)
                        for t in actual
                    ]
                    actual_grads = torch.autograd.grad(actual, (pts1, pts2), upstream)
                    expected_grads = torch.autograd.grad(
                        expected, (pts1, pts2), upstream
                    )
                    for a, b in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(a, b)

    def test_one_implicit_view_does_not_bias_joint_scale(self):
        pts1 = torch.tensor([[[[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]]]], dtype=torch.float64)
        pts2 = torch.tensor(
            [[[[2.0, 4.0, 4.0], [20.0, 40.0, 40.0]]]], dtype=torch.float64
        )
        valid2 = torch.tensor([[[True, False]]])
        actual = normalize_pointcloud(pts1, pts2, valid2=valid2, ret_factor=True)
        # Two unmasked points at distance3 and one valid point at distance6.
        expected_factor = torch.tensor([[[[12.0 / (3 + 1e-8)]]]], dtype=torch.float64)
        torch.testing.assert_close(actual[-1], expected_factor)
        torch.testing.assert_close(actual[0], pts1 / expected_factor)
        torch.testing.assert_close(actual[1], pts2 / expected_factor)

    def test_single_view_matches_explicit_mask(self):
        generator = torch.Generator().manual_seed(11)
        pts = torch.randn(2, 3, 4, 3, generator=generator, dtype=torch.float64)
        actual = normalize_pointcloud(pts, None)
        expected = normalize_pointcloud(
            pts, None, valid1=torch.ones(pts.shape[:-1], dtype=torch.bool)
        )
        torch.testing.assert_close(actual, expected)

    def test_unmasked_normalization_gradcheck(self):
        generator = torch.Generator().manual_seed(17)
        pts = torch.randn(
            1, 2, 3, generator=generator, dtype=torch.float64
        ).requires_grad_()
        self.assertTrue(
            torch.autograd.gradcheck(lambda p: normalize_pointcloud(p, None), (pts,))
        )

    def test_empty_batch_count(self):
        actual, count = invalid_to_zeros(torch.empty(0, 3, 4, 3), None, ndim=3)
        self.assertEqual(count, 0)
        self.assertEqual(actual.shape, (0, 12, 3))


if __name__ == "__main__":
    unittest.main()
