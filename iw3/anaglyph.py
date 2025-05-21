import torch


def grayscale_bt601(x, num_output_channels=1):
    y = x[0:1] * 0.299 + x[1:2] * 0.587 + x[2:3] * 0.114
    return torch.cat([y for _ in range(num_output_channels)], dim=0)


def color(left_eye, right_eye):
    anaglyph = torch.cat((left_eye[0:1, :, :], right_eye[1:3, :, :]), dim=0)
    return anaglyph


def half_color(left_eye, right_eye):
    anaglyph = torch.cat((grayscale_bt601(left_eye, num_output_channels=1),
                          right_eye[1:3, :, :]), dim=0)
    anaglyph = torch.clamp(anaglyph, 0, 1)
    return anaglyph


def gray(left_eye, right_eye):
    ly = grayscale_bt601(left_eye, num_output_channels=3)
    ry = grayscale_bt601(right_eye, num_output_channels=3)
    anaglyph = torch.cat((ly[0:1, :, :], ry[1:3, :, :]), dim=0)
    anaglyph = torch.clamp(anaglyph, 0, 1)
    return anaglyph


def wimmer(left_eye, right_eye):
    # Wimmer's Optimized Anaglyph
    # https://3dtv.at/Knowhow/AnaglyphComparison_en.aspx
    anaglyph = torch.cat((left_eye[1:2, :, :] * 0.7 + left_eye[2:3, :, :] * 0.3,
                          right_eye[1:3, :, :]), dim=0)
    anaglyph = torch.clamp(anaglyph, 0, 1)
    return anaglyph


def wimmer2(left_eye, right_eye):
    # Wimmer's improved method
    # described in "Methods for computing color anaglyphs"
    g_l = left_eye[1:2] + 0.45 * torch.clamp(left_eye[0:1] - left_eye[1:2], min=0)
    b_l = left_eye[2:3] + 0.25 * torch.clamp(left_eye[0:1] - left_eye[2:3], min=0)
    g_r = right_eye[1:2] + 0.45 * torch.clamp(right_eye[0:1] - right_eye[1:2], min=0)
    b_r = right_eye[2:3] + 0.25 * torch.clamp(right_eye[0:1] - right_eye[2:3], min=0)
    left = (0.75 * g_l + 0.25 * b_l) ** (1.0 / 1.6)
    anaglyph = torch.cat((left, g_r, b_r), dim=0)
    anaglyph = torch.clamp(anaglyph, 0, 1)
    return anaglyph


def dubois(left_eye, right_eye, clip_before=True):
    # Dubois method
    # reference: https://www.site.uottawa.ca/~edubois/anaglyph/LeastSquaresHowToPhotoshop.pdf
    def to_linear(x):
        cond1 = x <= 0.04045
        cond2 = torch.logical_not(cond1)
        x[cond1] = x[cond1] / 12.92
        x[cond2] = ((x[cond2] + 0.055) / 1.055) ** 2.4
        return x

    def to_nonlinear(x):
        cond1 = x <= 0.0031308
        cond2 = torch.logical_not(cond1)
        x[cond1] = x[cond1] * 12.92
        x[cond2] = 1.055 * x[cond2] ** (1.0 / 2.4) - 0.055
        return x

    def dot_clip(x, vec, clip):
        x = (x * vec).sum(dim=0, keepdim=True)
        if clip:
            x = x.clamp(0, 1)
        return x

    left_eye = to_linear(left_eye.detach().clone())
    right_eye = to_linear(right_eye.detach().clone())
    l_mat = torch.tensor([[0.437, 0.449, 0.164],
                          [-0.062, -0.062, -0.024],
                          [-0.048, -0.050, -0.017]],
                         device=left_eye.device, dtype=torch.float32).reshape(3, 3, 1, 1)
    r_mat = torch.tensor([[-0.011, -0.032, -0.007],
                          [0.377, 0.761, 0.009],
                          [-0.026, -0.093, 1.234]],
                         device=right_eye.device, dtype=torch.float32).reshape(3, 3, 1, 1)
    anaglyph = torch.cat([
        dot_clip(left_eye, l_mat[0], clip_before) + dot_clip(right_eye, r_mat[0], clip_before),
        dot_clip(left_eye, l_mat[1], clip_before) + dot_clip(right_eye, r_mat[1], clip_before),
        dot_clip(left_eye, l_mat[2], clip_before) + dot_clip(right_eye, r_mat[2], clip_before),
    ], dim=0)
    anaglyph = torch.clamp(anaglyph, 0, 1)
    anaglyph = to_nonlinear(anaglyph)
    anaglyph = torch.clamp(anaglyph, 0, 1)
    return anaglyph


def apply_anaglyph_redcyan(left_eye, right_eye, anaglyph_type):
    """
    根据指定的算法类型生成红青立体图像（Anaglyph 3D），通过将左右眼图像的颜色通道分离并重新组合，生成适合红蓝/红青眼镜观看的3D效果

    +----------------+-------------------------------------------------------------------+------------------------------------------------------------+
    |    参数类型    |                              处理方式                             |                          视觉效果                          |
    +================+===================================================================+============================================================+
    |      color     |     右眼保留绿色+蓝色通道（青），左眼保留红色通道，直接叠加。     |             色彩鲜艳但色差明显，可能产生重影。             |
    +----------------+-------------------------------------------------------------------+------------------------------------------------------------+
    |      gray      |   左右眼图像转为灰度图，再分别用红色（左）和青色（右）通道叠加。  |      减少颜色干扰，适合文本或黑白内容，立体感更清晰。      |
    +----------------+-------------------------------------------------------------------+------------------------------------------------------------+
    |   half-color   |     右眼保留全彩，左眼仅红色通道（或反之），平衡色彩与立体感。    |           保留部分色彩，减少重影，适合自然场景。           |
    +----------------+-------------------------------------------------------------------+------------------------------------------------------------+
    | wimmer/wimmer2 |        基于 Wimmer优化算法，通过非线性混合减少色差和鬼影。        |             更自然的色彩过渡，立体效果更稳定。             |
    +----------------+-------------------------------------------------------------------+------------------------------------------------------------+
    | dubois/dubois2 | 使用 Dubois色彩校正矩阵，在频域分离左右眼信号，最大化色彩保真度。 | 专业级效果，颜色失真最小，细节保留最佳（需高质量源图像）。 |
    +----------------+-------------------------------------------------------------------+------------------------------------------------------------+
    """
    if anaglyph_type == "color":
        return color(left_eye, right_eye)
    elif anaglyph_type == "gray":
        return gray(left_eye, right_eye)
    elif anaglyph_type == "half-color":
        return half_color(left_eye, right_eye)
    elif anaglyph_type == "wimmer":
        return wimmer(left_eye, right_eye)
    elif anaglyph_type == "wimmer2":
        return wimmer2(left_eye, right_eye)
    elif anaglyph_type in {"dubois", "dubois2"}:
        clip_before = True if anaglyph_type == "dubois" else False
        return dubois(left_eye, right_eye, clip_before=clip_before)
    else:
        raise ValueError(f"Unknown anaglyph_type {anaglyph_type}")
