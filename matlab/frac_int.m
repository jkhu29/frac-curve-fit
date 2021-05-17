function Iy = frac_int(x, y, v, M) %自变量 因变量 分数阶 精度倒数
% x y 均为数组，v为正实数，M为正整数
h = 1/M;
y1 = 0*x;
Iy = 0*x;

Iy(2) = y(2)  / 2 / v * h^(1+v);
for i=3:1:M+1  %分数阶积分
    y1(2:i-1) = ( ( ones(1,i-2).*i.*h - linspace(2,i-1,i-2).*h ).^(v-1).*y(2:i-1) + (ones(1,i-2).*i.*h - linspace(1,i-2,i-2).*h).^(v-1).*y(1:i-2) ) ./2 .*h;
    y1(i) = ( y(i) + y(i-1) ) / 2 / v * h^(1+v);
    Iy(i) = sum(y1);
end
Iy = Iy / gamma(v);

end