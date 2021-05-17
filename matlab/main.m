%% 数据输入
n = input("please input your n: ");
x = get_test_data(1, 1, 1, 1, 1, n);
t = 0:1/(n-1):1;

%% 计算数据图像的盒函数
% init
img = imread('data.png');
canny = edge(rgb2gray(img), 'canny');
% 设置小盒子的取值
e = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
% 判断矩阵大小
size_img = size(canny);
% 记录log(N)和log(1/ε)
log_n = zeros(size(e));
log_e = zeros(size(e));
% main
for l = 1:length(e)
    k = e(l);
    A = zeros(size_img-k+1);
    a = ones(k, k);
    for i = 1:k:size_img(1)-k+1
        for j = 1:k:size_img(2)-k+1
            A(i, j) = sum(canny(i:i+k-1, j:j+k-1) .* a, 'all');
        end
    end
    n_1 = sum(A ~= 0, 'all');
    log_n(l) = log(n_1);
    log_e(l) = log(1/k);
end
[fitresult1, gof1] = createFit_box_count(log_e, log_n); % 对数据进行线性拟合
box_counting = fitresult1.p1; % 得到盒维数
disp(box_counting); 

%% 五次多项式插值
y = zeros(1, n);
syms x1;
for i = 1:n
    f = @(x1) x(i, 2)*(t(i) - x1).^(box_counting-2);
    y(i) = int(f, x1, t(i), 0) / gamma(10.5); % todo
end
[fitresult2, gof2] = createFit_cn(t, y);
c1 = [fitresult2.e, fitresult2.d, fitresult2.c, fitresult2.b, fitresult2.a];
c2 = zeros(size(c1));
for i = 1:5
    c2(i) = gamma(i+1) * c1(i) / gamma(i+2-box_counting);
end

%% 画图检验
f_a = @(x1) (c2(1)*x1 + c2(2)*x1.^2 + c2(3)*x1.^3 + c2(4)*x1.^4 + c2(5)*x1.^5) ./ x1.^(box_counting-1);
fplot(f_a);
