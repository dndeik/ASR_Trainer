import math

import torch


def get_mask_cycle(chunk_size, mask_length, left_context=0, right_context=0):
    # округляем длину под чанки
    corrected_mask_length = math.ceil(mask_length / chunk_size) * chunk_size

    # общая ширина с учетом левого и правого контекста
    total_width = corrected_mask_length + left_context + right_context

    mask = torch.zeros(corrected_mask_length, total_width)

    for i in range(0, corrected_mask_length, chunk_size):
        row_start = i
        row_end = i + chunk_size

        col_start = i
        col_end = i + chunk_size + left_context + right_context

        mask[row_start:row_end, col_start:col_end] = 1.0

    # обрезаем до реальной длины
    mask = mask[:mask_length, left_context:left_context + mask_length]

    # переводим в attention mask
    mask = ~mask.bool() * float("-inf")
    mask = torch.nan_to_num(mask, neginf=float("-inf"))

    return mask


def get_mask_vector(chunk_size, mask_length, left_context=0, right_context=0, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Выравниваем размер
    corrected_length = (mask_length + chunk_size - 1) // chunk_size * chunk_size

    # 2. Матрицы индексов
    row_idx = torch.arange(corrected_length, device=device).unsqueeze(1)  # [L, 1]
    col_idx = torch.arange(
        corrected_length + left_context + right_context,
        device=device
    ).unsqueeze(0)  # [1, L+LC+RC]

    # 3. Индекс чанка
    row_chunk = row_idx // chunk_size  # [L, 1]

    # 4. Границы допустимого внимания
    start_col = row_chunk * chunk_size  # [L, 1]
    end_col = start_col + chunk_size + left_context + right_context

    # 5. Маска допустимых позиций
    chunk_mask = (col_idx >= start_col) & (col_idx < end_col)  # [L, L+LC+RC]

    # 6. Обрезаем до реальной длины
    chunk_mask = chunk_mask[
        :mask_length,
        left_context:left_context + mask_length
    ]  # [mask_length, mask_length]

    # 7. Attention mask
    attn_mask = torch.where(chunk_mask, 0.0, float('-inf'))

    return attn_mask


def cut_masks_with_len_vector(mask, lens, left_context):
    B = lens.size(0)
    L = mask.size(0)  # размер базовой маски

    # Заполним маски -inf (замаскированные)
    batch_masks = torch.full((B, L, L), fill_value=-5e4, device=mask.device)

    # Индексы для строк и столбцов
    rows = torch.arange(L, device=mask.device).unsqueeze(0).expand(B, -1)  # (B, L)
    cols = torch.arange(L, device=mask.device).unsqueeze(0).expand(B, -1)  # (B, L)

    # Длины с учетом контекста
    max_len = (lens + left_context).unsqueeze(-1)  # (B, 1)

    # Булевы маски для строк и столбцов
    row_mask = rows < max_len  # (B, L)
    col_mask = cols < max_len  # (B, L)

    # Пересечение масок для двумерной маски
    valid_mask = row_mask.unsqueeze(2) & col_mask.unsqueeze(1)  # (B, L, L)

    # Расширяем базовую маску на батч
    mask_expanded = mask.unsqueeze(0).expand(B, -1, -1)  # (B, L, L)

    # Формируем итоговую маску с подблоками из базовой маски
    batch_masks = torch.where(valid_mask, mask_expanded, batch_masks)

    return batch_masks


def cut_masks_with_len_cycle(mask, lens, left_context):
    B, L = lens.shape
    masks = []
    for i in range(B):
        cur_masks = torch.full_like(mask, fill_value=-5e4, device=mask.device)
        cur_masks[:lens[i] + left_context, :lens[i] + left_context] = mask[
            :lens[i] + left_context, :lens[i] + left_context]
        masks.append(cur_masks)
    masks = torch.stack(masks)
    return masks
