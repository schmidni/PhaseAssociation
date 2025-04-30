
from pytorch_metric_learning import distances, losses, miners, reducers

from src.models import PrototypicalLoss


def metric_loss_triplets(epoch, num_epochs) \
        -> tuple[miners.BaseMiner, losses.BaseMetricLossFunction]:
    """
    Returns a tuple of the mining function and loss function to use for
    training.
    """
    distance = distances.CosineSimilarity()
    reducer = reducers.AvgNonZeroReducer()
    margin = 0.2 + (epoch * 0.4 / num_epochs)

    mining_func = miners.TripletMarginMiner(
        margin=margin, distance=distance, type_of_triplets='semihard')
    loss_func = losses.TripletMarginLoss(
        margin=margin, distance=distance, reducer=reducer)

    return mining_func, loss_func


def metric_loss_multi_similarity(epoch, num_epochs) \
        -> tuple[miners.BaseMiner, losses.BaseMetricLossFunction]:
    """
    Returns a tuple of the mining function and loss function to use for
    training.
    """
    initial_eps = 0.2
    final_eps = 0.05
    epsilon = initial_eps + (final_eps - initial_eps) * (epoch / num_epochs)

    # Optionally scale α and β (positive/negative strength)
    alpha = 2 + 2 * (epoch / num_epochs)  # range from 2 → 4
    beta = 50 + 50 * (epoch / num_epochs)  # range from 50 → 100

    miner = miners.MultiSimilarityMiner(epsilon=epsilon)
    loss_func = losses.MultiSimilarityLoss(alpha=alpha, beta=beta, base=0.5)

    return miner, loss_func


def metric_loss_contrastive(epoch, num_epochs) \
        -> tuple[miners.BaseMiner, losses.BaseMetricLossFunction]:
    """
    Returns a tuple of the mining function and loss function to use for
    training.
    """
    distance = distances.LpDistance()
    margin = 0.2 + (epoch * 0.4/num_epochs)

    mining_func = miners.TripletMarginMiner(
        margin=margin, distance=distance, type_of_triplets='semihard')
    loss_func = losses.ContrastiveLoss()

    return mining_func, loss_func


def metric_loss_circle(epoch, num_epochs) \
        -> tuple[None, losses.CircleLoss]:
    """
    Returns an adaptive CircleLoss with epoch-dependent margin and gamma.
    """

    # Schedule margin: start loose (0.4) → tighter (0.25)
    m_start = 0.4
    m_end = 0.25
    m = m_start + (m_end - m_start) * (epoch / num_epochs)

    # Schedule gamma: start low (e.g. 40) → sharp decision boundary (80–100)
    gamma_start = 40
    gamma_end = 80
    gamma = gamma_start + (gamma_end - gamma_start) * (epoch / num_epochs)

    def mining_func(x, y): return None  # CircleLoss uses all valid pairs

    loss_func = losses.CircleLoss(m=m, gamma=gamma)

    return mining_func, loss_func


def metric_loss_prototypical(epoch, num_epochs) \
        -> tuple[miners.BaseMiner, losses.BaseMetricLossFunction]:
    """
    Returns a tuple of the mining function and loss function to use for
    training.
    """
    def mining_func(x, y): return None
    loss_func = PrototypicalLoss(distance='euclidean')

    return mining_func, loss_func
