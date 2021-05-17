function Wrs = wrs_fun(x,a,b,N) %自变量 参数lambda 参数alpha 迭代次数

m = 0*x;
Wrs = 0*x;

for n=1:1:N  %生成函数
m=a^(-b*n)*sin(a^(n)*x);
Wrs=Wrs+m;
end

end