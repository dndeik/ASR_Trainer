import torch

  
def rnnt_greedy_decode_batch(logits, blank=0):
    """
    logits: Tensor [B, T, U, V] - выход модели
    blank: индекс blank токена
    Returns:
    list из списков токенов для каждого примера в батче
    """

    B, T, U, V = logits.shape
    device = logits.device

    # Начальные позиции для каждого примера
    t_pos = torch.zeros(B, dtype=torch.int64, device=device)
    u_pos = torch.zeros(B, dtype=torch.int64, device=device)

    # Для каждого примера храним список предсказанных токенов
    results = [[] for _ in range(B)]

    # Флаг завершения для каждого примера
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    while (~finished).any():
        # Для всех не завершенных берем логиты на текущих позициях
        batch_indices = torch.arange(B, device=device)[~finished]

        t_indices = t_pos[~finished]
        u_indices = u_pos[~finished]

        # logits для текущих позиций: [active_batch, V]
        current_logits = logits[batch_indices, t_indices, u_indices, :]

        # Выбираем argmax токенов
        preds = torch.argmax(current_logits, dim=-1)

        # Обрабатываем выбор для каждого активного примера
        for i, b_idx in enumerate(batch_indices):
            if preds[i].item() == blank:
                # Если blank — сдвигаем t
                t_pos[b_idx] += 1
                # Если вышли за предел T — считаем пример завершенным
                if t_pos[b_idx] >= T:
                    finished[b_idx] = True
            else:
                # Добавляем токен в результат
                results[b_idx].append(preds[i].item())
                u_pos[b_idx] += 1
                # Если вышли за предел U — считаем пример завершенным
                if u_pos[b_idx] >= U:
                    finished[b_idx] = True

        # Если все примеры закончили, выход из цикла
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