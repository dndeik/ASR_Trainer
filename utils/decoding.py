import torch


def rnnt_greedy_decode_batch(
        logits,
        blank=0,
        max_symbols_per_frame=3,
):
    """
    logits: Tensor [B, T, U, V]
    blank: индекс blank токена
    max_symbols_per_frame: максимум небланковых токенов на один фрейм
    Returns:
    list из списков токенов для каждого примера в батче
    """

    B, T, U, V = logits.shape
    device = logits.device

    t_pos = torch.zeros(B, dtype=torch.int64, device=device)
    u_pos = torch.zeros(B, dtype=torch.int64, device=device)

    # Счётчик токенов на текущем фрейме
    symbols_emitted = torch.zeros(B, dtype=torch.int64, device=device)

    results = [[] for _ in range(B)]
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    while (~finished).any():
        batch_indices = torch.arange(B, device=device)[~finished]

        t_indices = t_pos[~finished]
        u_indices = u_pos[~finished]

        current_logits = logits[batch_indices, t_indices, u_indices, :]
        preds = torch.argmax(current_logits, dim=-1)

        for i, b_idx in enumerate(batch_indices):
            if finished[b_idx]:
                continue

            pred = preds[i].item()

            if pred == blank:
                # blank → двигаемся по времени
                t_pos[b_idx] += 1
                symbols_emitted[b_idx] = 0

                if t_pos[b_idx] >= T:
                    finished[b_idx] = True
            else:
                # эмиссия токена
                results[b_idx].append(pred)
                u_pos[b_idx] += 1
                symbols_emitted[b_idx] += 1

                if u_pos[b_idx] >= U:
                    finished[b_idx] = True
                    continue

                # если превысили лимит токенов на фрейм — форсим t++
                if symbols_emitted[b_idx] >= max_symbols_per_frame:
                    t_pos[b_idx] += 1
                    symbols_emitted[b_idx] = 0

                    if t_pos[b_idx] >= T:
                        finished[b_idx] = True

        if finished.all():
            break

    return results


def ctc_greedy_decode_batch(esti: torch.Tensor, blank: int):
    # 1. Создаем маску: токен != blank_token
    mask_not_blank = esti != blank  # shape [B, T]

    # 2. Создаем маску для удаления повторяющихся подряд токенов
    # Сдвинем по времени на 1 (влево) для сравнения с предыдущим токеном
    shifted = torch.zeros_like(esti)
    shifted[:, 1:] = esti[:, :-1]

    # Маска для выбора токенов, которые не равны предыдущему токену
    mask_no_repeat = esti != shifted  # shape [B, T]

    # Итоговая маска — токен не blank И не повторяется подряд
    mask = mask_not_blank & mask_no_repeat

    results = []
    for b in range(esti.size(0)):
        tokens = esti[b][mask[b]].cpu().tolist()
        results.append(tokens)

    return results