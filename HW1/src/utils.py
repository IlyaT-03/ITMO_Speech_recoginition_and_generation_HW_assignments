import torch


def count_parameters(model):
    """
    Подсчёт числа обучаемых параметров модели
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy_from_logits(logits, targets):
    """
    Вычисление accuracy из logits
    """
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def count_flops(model, device="cpu"):
    """
    Подсчёт FLOPs модели.

    Требует библиотеку thop:
        pip install thop
    """

    try:
        from thop import profile

        dummy_input = torch.randn(1, 16000).to(device)

        macs, _ = profile(
            model,
            inputs=(dummy_input,),
            verbose=False
        )

        # FLOPs = 2 * MACs
        flops = 2 * macs

        return flops

    except Exception:
        return None