"""Reproducible FastICA/PCA comparison on synthetic source signals."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA, PCA


RANDOM_SEED = 42


def standardize_columns(values):
    return (values - values.mean(axis=0)) / values.std(axis=0)


def align_components(reference, estimated):
    """Resolve ICA/PCA permutation, sign and scale ambiguities."""
    reference = standardize_columns(reference)
    estimated = standardize_columns(estimated)
    n_sources = reference.shape[1]
    correlation = np.corrcoef(reference.T, estimated.T)[:n_sources, n_sources:]
    source_indices, component_indices = linear_sum_assignment(-np.abs(correlation))

    aligned = np.empty_like(estimated)
    scores = np.empty(n_sources, dtype=float)
    for source_index, component_index in zip(source_indices, component_indices):
        sign = 1.0 if correlation[source_index, component_index] >= 0 else -1.0
        aligned[:, source_index] = sign * estimated[:, component_index]
        scores[source_index] = abs(correlation[source_index, component_index])
    return aligned, scores, correlation


def generate_sources(n_samples, rng):
    """Create four non-Gaussian sources without deterministic duplication."""
    time = np.linspace(0, 10, n_samples, endpoint=False)
    base_frequency = 0.5
    sources = np.column_stack(
        [
            np.sin(2 * np.pi * base_frequency * time - np.pi / 4),
            np.sign(
                np.sin(2 * np.pi * base_frequency * np.sqrt(2) * time - np.pi / 5)
            ),
            signal.sawtooth(2 * np.pi * base_frequency * np.sqrt(5) * time),
            rng.laplace(size=n_samples),
        ]
    )
    return time, standardize_columns(sources)


def run_experiment(n_samples=10000, seed=RANDOM_SEED, plot=True):
    rng = np.random.default_rng(seed)
    time, sources = generate_sources(n_samples, rng)
    mixing = rng.standard_normal((sources.shape[1], sources.shape[1]))
    mixed = sources @ mixing.T

    ica = FastICA(
        n_components=sources.shape[1],
        algorithm='deflation',
        whiten='unit-variance',
        max_iter=2000,
        tol=1e-5,
        random_state=seed,
    )
    estimated_ica = ica.fit_transform(mixed)
    estimated_pca = PCA(n_components=sources.shape[1]).fit_transform(mixed)

    aligned_ica, ica_scores, ica_correlation = align_components(
        sources,
        estimated_ica,
    )
    aligned_pca, pca_scores, pca_correlation = align_components(
        sources,
        estimated_pca,
    )

    print('Absolute source correlations after optimal alignment:')
    print(f'FastICA: {np.round(ica_scores, 4)}')
    print(f'PCA:     {np.round(pca_scores, 4)}')
    print(f'FastICA mean correlation: {ica_scores.mean():.4f}')
    print(f'PCA mean correlation:     {pca_scores.mean():.4f}')

    if plot:
        names = ['Sine', 'Square', 'Sawtooth', 'Laplace noise']
        display_count = min(n_samples, 2500)

        plt.figure(figsize=(10, 8))
        for index, name in enumerate(names):
            axis = plt.subplot(len(names), 1, index + 1)
            axis.plot(time[:display_count], sources[:display_count, index])
            axis.set_title(f'Original source: {name}')
            axis.grid()
        plt.tight_layout()

        for method_name, aligned in (
            ('FastICA', aligned_ica),
            ('PCA', aligned_pca),
        ):
            plt.figure(figsize=(10, 8))
            for index, name in enumerate(names):
                axis = plt.subplot(len(names), 1, index + 1)
                axis.plot(
                    time[:display_count],
                    sources[:display_count, index],
                    '--',
                    label='source',
                )
                axis.plot(
                    time[:display_count],
                    aligned[:display_count, index],
                    label=method_name,
                    alpha=0.8,
                )
                axis.set_title(name)
                axis.grid()
                axis.legend()
            plt.tight_layout()

        plt.show()

    return {
        'sources': sources,
        'mixed': mixed,
        'aligned_ica': aligned_ica,
        'aligned_pca': aligned_pca,
        'ica_correlations': ica_scores,
        'pca_correlations': pca_scores,
        'ica_correlation_matrix': ica_correlation,
        'pca_correlation_matrix': pca_correlation,
        'mixing_condition_number': float(np.linalg.cond(mixing)),
    }


if __name__ == '__main__':
    run_experiment()
