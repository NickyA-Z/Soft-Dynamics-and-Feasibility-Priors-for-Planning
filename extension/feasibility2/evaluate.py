"""
Checks if learned energy is meaningful
Checks if learned energy is meaningful
compares expert transitions against corrupted/shuffled transitions:
\mathcal{P}_\theta(z_{expert}) < \mathcal{P}_\theta(z_{corrupt})

"""

from __future__ import annotations

import argparse
from unittest import loader

import torch
from torch.utils.data import DataLoader

try:
    from .dataset import load_tensor_dataset
    from .model import FeasibilityModel, load_feasibility_model_from_checkpoint
except ImportError:
    from extension.feasibility2.dataset import load_tensor_dataset
    from extension.feasibility2.model import FeasibilityModel, load_feasibility_model_from_checkpoint

def print_metrics_section(title: str, metrics: dict[str, float]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:55s}: {v:.6f}")
        else:
            print(f"{k:55s}: {v}")

@torch.no_grad()
def evaluate_expert_vs_random(
    model: FeasibilityModel,
    dataset,
    batch_size: int = 256,
    noise_level: float = 0.2,
    device: str | torch.device | None = None,
    random_mode: str = "shuffle",
) -> dict[str, float]:
    """Check whether expert next latents have lower learned energy than negatives.

    random_mode='shuffle' uses other z_next values from the dataset as negatives.
    random_mode='gaussian' samples N(mean,std) using dataset next-latent statistics.
    """
    device = torch.device(device or next(model.parameters()).device)
    model = model.to(device).eval()
    if noise_level is None:
        #noise_level = model.noise_level
        noise_level = getattr(model, "noise_level", 0.2)
    print(f"Using noise_level={noise_level}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # do these get used
    # do these get used
    z_mean = dataset.next_latents.mean(dim=0).to(device)
    z_std = dataset.next_latents.std(dim=0).clamp_min(1e-6).to(device)

    expert_energies = []
    random_energies = []
    cosines = []

    delta_mses = []
    zero_delta_mses = []
    transition_penalties = []


    for batch in loader:
        if len(batch) == 4:
            history, action, z_next, past_action_history = batch
        else:
            history, action, z_next = batch
            past_action_history = None       
    #for history, action, z_next in loader:
        history = history.to(device)
        action = action.to(device)
        z_next = z_next.to(device)
        if past_action_history is not None:
            past_action_history = past_action_history.to(device)

        if random_mode == "shuffle":
            perm = torch.randperm(z_next.shape[0], device=device)
            z_neg = z_next[perm]
        elif random_mode == "gaussian":
            z_neg = z_mean + torch.randn_like(z_next) * z_std
        elif random_mode == "small_noise":
            added_noise = torch.randn_like(z_next)
            corruption_scale = noise_level
            # corruption_scale = 0.1
            z_neg = z_next + corruption_scale * added_noise
            true_eps = (z_neg - z_next) / float(noise_level)

        else:
            raise ValueError("random_mode must be 'shuffle', 'gaussian', or 'small_noise'")
        
        if len(expert_energies) == 0:
            print("\n[expert_vs_random debug]")
            print("DEBUG z_next shape:", tuple(z_next.shape))
            print("DEBUG z_neg shape:", tuple(z_neg.shape))
            print("DEBUG z diff mean:", float((z_next - z_neg).abs().mean().cpu()))
            print("DEBUG z diff max:", float((z_next - z_neg).abs().max().cpu()))


        e_exp = model.penalty(history, action, z_next, noise_level=noise_level, reduction="none")
        e_neg = model.penalty(history, action, z_neg, noise_level=noise_level, reduction="none")
        expert_energies.append(e_exp.detach().cpu())
        random_energies.append(e_neg.detach().cpu())

        if random_mode == "small_noise":
            pred_noise = model.forward(
                history,
                action,
                z_neg,
                noise_level=torch.as_tensor(noise_level, device=device),
            )
            '''
            cos = torch.nn.functional.cosine_similarity(
                pred_noise.reshape(pred_noise.shape[0], -1),
                added_noise.reshape(added_noise.shape[0], -1),
                dim=-1,
            ) '''
            #ToDO look into this
            cos = torch.nn.functional.cosine_similarity(
                pred_noise.reshape(pred_noise.shape[0], -1),
                true_eps.reshape(true_eps.shape[0], -1),
                dim=-1,
            )
            cosines.append(cos.detach().cpu())

            if len(expert_energies) == 0:
                print("\n[small_noise debug]")
                print("DEBUG cosine first 5:", cos[:5].detach().cpu())
                print("DEBUG cosine mean batch:", float(cos.mean().cpu()))
                print("z_next std:", z_next.std().item())
                print("noise std:", (0.1 * torch.randn_like(z_next)).std().item())
                print("noise std small:", (corruption_scale * added_noise).std().item())
                print("diff mean:", (z_next - z_neg).abs().mean().item())

    
        if hasattr(model, "predict_delta"):
            pred_delta = model.predict_delta(history, action)
            target_delta = z_next - history[:, -1]

            delta_mse = (pred_delta - target_delta).pow(2).reshape(history.shape[0], -1).mean(dim=-1)
            zero_delta_mse = target_delta.pow(2).reshape(history.shape[0], -1).mean(dim=-1)

            delta_mses.append(delta_mse.detach().cpu())
            zero_delta_mses.append(zero_delta_mse.detach().cpu())

            if hasattr(model, "transition_penalty"):
                trans = model.transition_penalty(
                    history,
                    action,
                    z_next,
                    reduction="none",
                )
                transition_penalties.append(trans.detach().cpu())
    
    expert = torch.cat(expert_energies)
    random = torch.cat(random_energies)
    margin = random - expert

    metrics = {
        "expert_energy_mean": float(expert.mean()),
        "expert_energy_median": float(expert.median()),
        "random_energy_mean": float(random.mean()),
        "random_energy_median": float(random.median()),
        "margin_mean_random_minus_expert": float(margin.mean()),
        "fraction_expert_lower_than_random": float((expert < random).float().mean()),
        "num_samples": int(expert.numel()),
    }

    if cosines:
        cos_all = torch.cat(cosines)
        metrics["cosine_mean_pred_vs_true_noise"] = float(cos_all.mean())
        metrics["cosine_median_pred_vs_true_noise"] = float(cos_all.median())
    
    if delta_mses:
        delta_mse_all = torch.cat(delta_mses)
        zero_delta_mse_all = torch.cat(zero_delta_mses)

        metrics["delta_mse_mean"] = float(delta_mse_all.mean())
        metrics["zero_delta_mse_mean"] = float(zero_delta_mse_all.mean())
        metrics["delta_vs_zero_ratio"] = float(delta_mse_all.mean() / zero_delta_mse_all.mean().clamp_min(1e-12))

        if transition_penalties:
            trans_all = torch.cat(transition_penalties)
            metrics["transition_penalty_expert_mean"] = float(trans_all.mean())

    return metrics

@torch.no_grad()
def evaluate_action_discrimination(
    model: FeasibilityModel,
    dataset,
    batch_size: int = 256,
    device: str | torch.device | None = None,
    gaussian_action_std: float = 1.0,
) -> dict[str, float]:
    """verifies whether feasibility transition/DSM prefers expert actions over wrong actions.
    This tests the planning-relevant inverse question:
        fixed history + fixed z_next:
            is transition_penalty(history, expert_action, z_next)
            lower than transition_penalty(history, wrong_action, z_next)?
    Wrong actions tested:
        - zero action
        - shuffled action from another sample in the batch
        - sign-flipped expert action
        - Gaussian random action
    """
    if not hasattr(model, "transition_penalty"):
        raise ValueError("Model does not have transition_penalty; train/load a delta-head model first.")
    device = torch.device(device or next(model.parameters()).device)
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    expert_penalties = []
    zero_penalties = []
    shuffle_penalties = []
    neg_penalties = []
    gaussian_penalties = []
    dsm_expert_actions = []
    dsm_zero_actions = []
    dsm_shuffle_actions = []
    dsm_neg_actions = []
    dsm_gaussian_actions = []
    for batch in loader:
        if len(batch) == 4:
            history, action, z_next, _ = batch
        else:
            history, action, z_next = batch

        history = history.to(device)
        action = action.to(device)
        z_next = z_next.to(device)
        zero_action = torch.zeros_like(action)
        perm = torch.randperm(action.shape[0], device=device)
        shuffle_action = action[perm]
        neg_action = -action
        gaussian_action = torch.randn_like(action) * gaussian_action_std
        trans_expert = model.transition_penalty(history, action, z_next, reduction="none")
        trans_zero = model.transition_penalty(history, zero_action, z_next, reduction="none")
        trans_shuffle = model.transition_penalty(history, shuffle_action, z_next, reduction="none")
        trans_neg = model.transition_penalty(history, neg_action, z_next, reduction="none")
        trans_gaussian = model.transition_penalty(history, gaussian_action, z_next, reduction="none")
        expert_penalties.append(trans_expert.detach().cpu())
        zero_penalties.append(trans_zero.detach().cpu())
        shuffle_penalties.append(trans_shuffle.detach().cpu())
        neg_penalties.append(trans_neg.detach().cpu())
        gaussian_penalties.append(trans_gaussian.detach().cpu())
        # Optional DSM/action sensitivity check.
        # This uses the same z_next but swaps only the action.
        if hasattr(model, "penalty"):
            noise_level = getattr(model, "noise_level", 0.2)
            dsm_expert_actions.append(
                model.penalty(history, action, z_next, noise_level=noise_level, reduction="none").detach().cpu()
            )
            dsm_zero_actions.append(
                model.penalty(history, zero_action, z_next, noise_level=noise_level, reduction="none").detach().cpu()
            )
            dsm_shuffle_actions.append(
                model.penalty(history, shuffle_action, z_next, noise_level=noise_level, reduction="none").detach().cpu()
            )
            dsm_neg_actions.append(
                model.penalty(history, neg_action, z_next, noise_level=noise_level, reduction="none").detach().cpu()
            )
            dsm_gaussian_actions.append(
                model.penalty(history, gaussian_action, z_next, noise_level=noise_level, reduction="none").detach().cpu()
            )
    expert = torch.cat(expert_penalties)
    zero = torch.cat(zero_penalties)
    shuffle = torch.cat(shuffle_penalties)
    neg = torch.cat(neg_penalties)
    gaussian = torch.cat(gaussian_penalties)
    metrics = {
        "transition_expert_action_mean": float(expert.mean()),
        "transition_expert_action_median": float(expert.median()),
        "transition_zero_action_mean": float(zero.mean()),
        "transition_shuffle_action_mean": float(shuffle.mean()),
        "transition_neg_action_mean": float(neg.mean()),
        "transition_gaussian_action_mean": float(gaussian.mean()),
        "margin_zero_minus_expert": float((zero - expert).mean()),
        "margin_shuffle_minus_expert": float((shuffle - expert).mean()),
        "margin_neg_minus_expert": float((neg - expert).mean()),
        "margin_gaussian_minus_expert": float((gaussian - expert).mean()),
        "fraction_expert_better_than_zero_action": float((expert < zero).float().mean()),
        "fraction_expert_better_than_shuffle_action": float((expert < shuffle).float().mean()),
        "fraction_expert_better_than_neg_action": float((expert < neg).float().mean()),
        "fraction_expert_better_than_gaussian_action": float((expert < gaussian).float().mean()),
        "num_samples": int(expert.numel()),
    }
    if dsm_expert_actions:
        dsm_expert = torch.cat(dsm_expert_actions)
        dsm_zero = torch.cat(dsm_zero_actions)
        dsm_shuffle = torch.cat(dsm_shuffle_actions)
        dsm_neg = torch.cat(dsm_neg_actions)
        dsm_gaussian = torch.cat(dsm_gaussian_actions)
        metrics.update(
            {
                "dsm_expert_action_mean": float(dsm_expert.mean()),
                "dsm_zero_action_mean": float(dsm_zero.mean()),
                "dsm_shuffle_action_mean": float(dsm_shuffle.mean()),
                "dsm_neg_action_mean": float(dsm_neg.mean()),
                "dsm_gaussian_action_mean": float(dsm_gaussian.mean()),
                "dsm_fraction_expert_better_than_zero_action": float((dsm_expert < dsm_zero).float().mean()),
                "dsm_fraction_expert_better_than_shuffle_action": float((dsm_expert < dsm_shuffle).float().mean()),
                "dsm_fraction_expert_better_than_neg_action": float((dsm_expert < dsm_neg).float().mean()),
                "dsm_fraction_expert_better_than_gaussian_action": float((dsm_expert < dsm_gaussian).float().mean()),
            }
        )
    return metrics

def evaluate_action_conversion(dataset, metadata) -> dict[str, float]:
    """verifies action scale/convention."""
    actions = dataset.actions.float()

    print("dataset action shape:", tuple(actions.shape))
    print("normalized action mean:", actions.mean(dim=0))
    print("normalized action std:", actions.std(dim=0))
    print("normalized action min:", actions.min(dim=0).values)
    print("normalized action max:", actions.max(dim=0).values)

    action_mean = None
    action_std = None

    if isinstance(metadata, dict):
        action_mean = metadata.get("action_mean", None)
        action_std = metadata.get("action_std", None)

    if action_mean is None or action_std is None:
        print("No action_mean/action_std found in metadata.")
        return {}

    action_mean = torch.as_tensor(action_mean).float()
    action_std = torch.as_tensor(action_std).float()

    #raw_actions = (actions * action_std + action_mean) * 100.0
    action_mean = action_mean.to(actions.device)
    action_std = action_std.to(actions.device)

    if actions.shape[-1] != action_mean.numel():
        assert actions.shape[-1] % action_mean.numel() == 0
        repeat = actions.shape[-1] // action_mean.numel()
        action_mean = action_mean.repeat(repeat)
        action_std = action_std.repeat(repeat)

    raw_actions = (actions * action_std + action_mean) * 100.0

    """
    print("expanded action_mean:", action_mean)
    print("expanded action_std:", action_std)
    print("raw action mean:", raw_actions.mean(dim=0))
    print("raw action std:", raw_actions.std(dim=0))
    print("raw action min:", raw_actions.min(dim=0).values)
    print("raw action max:", raw_actions.max(dim=0).values)
    print("first 5 normalized actions:", actions[:5])
    print("first 5 raw actions:", raw_actions[:5])
    """

    return {
        "normalized_action_absmean": float(actions.abs().mean()),
        "raw_action_absmean": float(raw_actions.abs().mean()),
        "normalized_action_mean_global": float(actions.mean()),
        "normalized_action_std_global": float(actions.std()),
        "raw_action_mean_global": float(raw_actions.mean()),
        "raw_action_std_global": float(raw_actions.std()),
        "raw_action_min_global": float(raw_actions.min()),
        "raw_action_max_global": float(raw_actions.max()),
    }

# eval con
@torch.no_grad()
def evaluate_contrastive_energy(
    model,
    dataset,
    batch_size: int = 256,
    noise_level: float = 0.2,
    margin: float = 0.1,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Evaluate the scalar energy head on wrong-action negatives.

    Positive:
        E(history, expert_action, z_next)

    Negative:
        E(history, wrong_action, z_next)

    Lower energy is expected for the positive transition.
    """
    if not hasattr(model, "energy"):
        raise ValueError(
            "This model has no scalar energy head. "
            "Load a two-head transformer checkpoint."
        )

    device = torch.device(
        device or next(model.parameters()).device
    )
    model = model.to(device).eval()

    if noise_level is None:
        noise_level = 0.2

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    positive_energies = []
    negative_energies = []

    for batch in loader:
        if len(batch) == 4:
            history, action, z_next, _ = batch
        else:
            history, action, z_next = batch

        history = history.to(device)
        action = action.to(device)
        z_next = z_next.to(device)

        # A singleton batch cannot provide a different in-batch action.
        if action.shape[0] < 2:
            continue

        # This matches the wrong-action construction used during training.
        wrong_action = action.roll(shifts=1, dims=0)

        sigma = torch.full(
            (action.shape[0],),
            float(noise_level),
            device=device,
            dtype=z_next.dtype,
        )

        # Evaluate positive and negative examples together, matching training.
        energies = model.energy(
            torch.cat((history, history), dim=0),
            torch.cat((action, wrong_action), dim=0),
            torch.cat((z_next, z_next), dim=0),
            noise_level=torch.cat((sigma, sigma), dim=0),
            reduction="none",
        )

        positive, negative = energies.chunk(2, dim=0)

        positive_energies.append(positive.cpu())
        negative_energies.append(negative.cpu())

    if not positive_energies:
        raise ValueError(
            "No contrastive pairs were available for evaluation."
        )

    positive = torch.cat(positive_energies)
    negative = torch.cat(negative_energies)
    gap = negative - positive

    return {
        "positive_energy_mean": float(positive.mean()),
        "positive_energy_median": float(positive.median()),
        "negative_energy_mean": float(negative.mean()),
        "negative_energy_median": float(negative.median()),
        "energy_gap_mean_negative_minus_positive": float(gap.mean()),
        "energy_gap_median_negative_minus_positive": float(gap.median()),
        "ranking_accuracy": float((gap > 0).float().mean()),
        "margin_accuracy": float((gap > margin).float().mean()),
        "pairwise_hinge_loss": float(
            torch.relu(margin - gap).mean()
        ),
        "num_pairs": int(gap.numel()),
    }

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate learned feasibility energies.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--noise-level", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--random-mode", choices=["shuffle", "gaussian", "small_noise"], default="shuffle")

    p.add_argument("--eval-mode", choices=["expert_vs_random", "action","contrastive", "both", "all"], default="expert_vs_random")
    p.add_argument("--gaussian-action-std", type=float, default=1.0)
    p.add_argument(
        "--check-action-conversion",
        action="store_true",
        help="Print normalized and reconstructed raw action statistics.",
    )
    p.add_argument(
        "--contrastive-margin",
        type=float,
        default=0.1,
    )

    return p.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset, metadata = load_tensor_dataset(args.dataset)
    #model = FeasibilityModel.from_checkpoint(args.checkpoint, map_location=device)
    model = load_feasibility_model_from_checkpoint(args.checkpoint, map_location=device)

    all_metrics: dict[str, float] = {}

    print("\n" + "=" * 80)
    print("FEASIBILITY EVALUATION")
    print("=" * 80)
    print(f"dataset:     {args.dataset}")
    print(f"checkpoint:  {args.checkpoint}")
    print(f"eval_mode:   {args.eval_mode}")
    print(f"device:      {device}")
    print(f"batch_size:  {args.batch_size}")

    if args.check_action_conversion:
        metrics = evaluate_action_conversion(dataset, metadata)
        all_metrics.update(metrics)
        print_metrics_section(
            "1. ACTION CONVERSION CHECK: normalized actions -> raw env scale",
            metrics,
        )

    if args.eval_mode in ("expert_vs_random", "both", "all"):
        metrics = evaluate_expert_vs_random(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            noise_level=args.noise_level,
            device=device,
            random_mode=args.random_mode,
        )
        print_metrics_section(
            f"2. DSM LATENT PLAUSIBILITY: expert z_next vs {args.random_mode} z_next",
            metrics,
        )

        #all_metrics.update(metrics)
    if args.eval_mode in ("action", "both", "all"):
        metrics = evaluate_action_discrimination(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            device=device,
            gaussian_action_std=args.gaussian_action_std,
        )
        print_metrics_section(
            "3. ACTION DISCRIMINATION: expert action vs wrong actions for same z_next",
            metrics,
        )
        #all_metrics.update(metrics)
        if args.eval_mode in ("contrastive", "all"):
            metrics = evaluate_contrastive_energy(
                model=model,
                dataset=dataset,
                batch_size=args.batch_size,
                noise_level=args.noise_level,
                margin=args.contrastive_margin,
                device=device,
            )

            print_metrics_section(
                "4. CONTRASTIVE ENERGY: expert action vs wrong action",
                metrics,
            )
    """
    for k, v in all_metrics.items():
        print(f"{k}: {v}")
"""

if __name__ == "__main__":
    main()
