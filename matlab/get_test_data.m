function output = get_test_data(a, b, c, d, e, n)
% 五次多项式 ,生成n个随机数，确保y(0) = 0
rand('seed', 118);
x = zeros(n, 2);
for i = 1:n
    x(i, 1) = 10 * rand;
    x(i, 2) = a*x(i, 1)^5+b*x(i, 1)^4+c*x(i, 1)^3+d*x(i, 1)^2+e*x(i, 1) + rand * 0.1;
end
figure;
plot(x(:, 1), x(:, 2), '.r');
figure;
plot(x(:, 1), x(:, 2), '.r');
set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w','position',[0 0 1 1]);
saveas(gcf, 'data', 'png');
output = x;
end