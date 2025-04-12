import numpy as np


def f1(grad: np.ndarray, kappa: float) -> np.ndarray:
    return np.exp(-((grad / kappa) ** 2))


# unused
def f2(grad: np.ndarray, kappa: float) -> np.ndarray:
    return 1 / (1 + (grad / kappa) ** 2)


def anisotropic(
    image: np.ndarray,
    iterations: int,
    kappa: float = 50.0,
    t_delta: float = 0.25,
) -> np.ndarray:
    image_f = np.copy(image).astype(float)

    for _ in range(iterations):
        dn = image_f[:-2, 1:-1] - image_f[1:-1, 1:-1]
        ds = image_f[2:, 1:-1] - image_f[1:-1, 1:-1]
        de = image_f[1:-1, 2:] - image_f[1:-1, 1:-1]
        dw = image_f[1:-1, :-2] - image_f[1:-1, 1:-1]

        diffs = np.stack([dn, ds, de, dw], axis=2)
        diffusivity = np.stack([f1(diffs[..., i], kappa) for i in range(diffs.shape[-1])], axis=2)

        image_f[1:-1, 1:-1] = image_f[1:-1, 1:-1] + t_delta * np.sum(diffs * diffusivity, axis=2)

    return image_f
