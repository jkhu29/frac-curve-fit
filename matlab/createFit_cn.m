function [fitresult, gof] = createFit_cn(t, y)
%CREATEFIT(T,Y)
%  Create a fit.
%
%  Data for 'fit_c' fit:
%      X Input : t
%      Y Output: y
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  另请参阅 FIT, CFIT, SFIT.

%  由 MATLAB 于 20-Jan-2021 14:52:17 自动生成


%% Fit: 'fit_c'.
[xData, yData] = prepareCurveData( t, y );

% Set up fittype and options.
ft = fittype( {'x^5', 'x^4', 'x^3', 'x^2', 'x'}, 'independent', 'x', 'dependent', 'y', 'coefficients', {'a', 'b', 'c', 'd', 'e'} );

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft );

% Plot fit with data.
figure( 'Name', 'fit_c' );
h = plot( fitresult, xData, yData );
legend( h, 'y vs. t', 'fit_c', 'Location', 'NorthEast', 'Interpreter', 'none' );
% Label axes
xlabel( 't', 'Interpreter', 'none' );
ylabel( 'y', 'Interpreter', 'none' );
grid on


