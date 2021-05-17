function Dy = frac_dif(x, y, u, M) %自变量 因变量 分数阶 精度倒数
% x y 均为数组，u为(0,1)间正实数，M为正整数
v = 1 - u;

Iy = frac_int(x, y, v, M);
Iy = diff(Iy);
Iy = Iy * M;
Dy = [0 Iy];




end