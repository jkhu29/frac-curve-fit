function [fitresult, gof] = createFit_box_count(log_e, log_n)
%CREATEFIT(LOG_E,LOG_N)
%  Create a fit.
%
%  Data for 'box_count' fit:
%      X Input : log_e
%      Y Output: log_n
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  另请参阅 FIT, CFIT, SFIT.

%  由 MATLAB 于 27-Nov-2020 13:10:04 自动生成


%% Fit: 'box_count'.
[xData, yData] = prepareCurveData( log_e, log_n );

% Set up fittype and options.
ft = fittype( 'poly1' );

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft );

% Plot fit with data.
figure( 'Name', 'box_count' );
h = plot( fitresult, xData, yData );
legend( h, 'log_n vs. log_e', 'box_count', 'Location', 'NorthEast', 'Interpreter', 'none' );
% Label axes
xlabel( 'log_e', 'Interpreter', 'none' );
ylabel( 'log_n', 'Interpreter', 'none' );
grid on


