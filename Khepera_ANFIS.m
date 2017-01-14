%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.
%    If not, see <http://www.gnu.org/licenses/gpl.txt>.

%    Version 1 Khepera_ANFIS.m coded by
%    Eric Nichols eric.nichols.phd@gmail.com

function [ output_args ] = Khepera_ANFIS( input_args )

% global variables that will be needed in both functions
global anfisOutput ref out_fis positive v FORWARD LEFT RIGHT trainingData

% Open our reference to the Khepera
ref = kopen([0, 9600, 100]);

FORWARD    =   0;    % 0==forward
LEFT       =  .2;    % .2==left
RIGHT      = -.2;    % -.2==right
ambientOLD = 500;    % brightest ambient lighting
epoch_n    = 20;     % number of epocs to train
trainingMatrix( 0 ); % get initial training data

% generate an initial fuzzy inference system with the
% training data and 3 trapezoid membership functions
in_fis = genfis1([trainingData], 3, 'trapmf');

% crete the adaptive neuro-fuzzy inference system
out_fis = anfis([trainingData], in_fis, epoch_n);

% this loop will continue for the
% duration of kepera's running
for i=1:300
    
    ambient = kAmbient(ref);                     % get the ambient lighting
    ambientAvg = ((ambient(7) + ambient(8)) /2); % avg for sensors
    
    % With a dark environment, Khepera will think it is
    % closer to an object. Therefore we want it to turn later.
    % The opposite is true for a brighter environment
    if (ambientAvg ~= ambientOLD) % lighting has changed
        
        % prepare for manipulation
        if (ambientAvg < 500)
            x = 500 - ambientAvg;          % difference from max (500)
            darker = (5.25*x*x) - (4.5*x); % formulae to make darker
        else                               % ambientAgv == 500
            darker = 0;                    % goto original settings
        end
        
        trainingMatrix( darker ); % manipulate inputs
        
        % re-train with updated data
        in_fis = genfis1([trainingData], 3, 'trapmf');
        out_fis = anfis([trainingData], in_fis, epoch_n);
        
        % this is so we'll know the current settings
        ambientOLD = ambientAvg;
    end % end of ambient light has changed
    
    v = kProximity(ref);  % read the proximity
    % evaluate the system based on the new
    % proximity readings to find Kep's classification
    anfisOutput = evalfis([(v(7) > v(8)) ((v(7) + v(8)) /2)], out_fis);
    
    % use the anfis output to find the course of action
    if (anfisOutput > LEFT)       % the object is Left
        turn(5, -5);              % turn right
    else
        if (anfisOutput < RIGHT)  % the object is right
            turn(-5, 5);          % turn left
        else                      % otherwise
            kSetSpeed(ref,-5,-5); % forward
        end
    end % end else if
end % end for

kSetSpeed(ref,0,0); % stop the Kepera
kclose(ref);        % close the reference to the Khepera
end

%This function is called to turn the Khepera robot
function [ ] = turn( lWheel, rWheel )
    global anfisOutput ref out_fis LEFT RIGHT; % global variables
    
     % move in the parametered direction
    kSetSpeed(ref, lWheel, rWheel);
    
    while ((anfisOutput > LEFT) || (anfisOutput < RIGHT)) % keep turning
        v = kProximity(ref); % read the proximity & evaluate it
        anfisOutput = evalfis([(v(7) > v(8)) ((v(7) + v(8)) /2)], out_fis);
    end
end

% These are the training values
% Positive means an object is closer to the left
% Negative means an object is closer to the right
% Zero means an object is not close
function [ ] = trainingMatrix( darker )
    global trainingData
    turnRange = [(darker+201):10:1021]'; % range for turning
    matrixLength = length(turnRange);    % store the length
    
    % positive value
    turnTrue = ones(matrixLength, 1);         % vecter for positive turns
    positive = [turnTrue turnRange turnTrue]; % positive matrix
    positive = [positive; 1 1024 1];          % add maximum possible value
    
    % negative value
    turnFalse = zeros(matrixLength, 1); % vecter for negative turns
    neg = turnFalse;                    % copy the vecter
    for w=1:matrixLength                % for every element
        neg(w) = -1;                    % change neg to negative 1
    end
    negative = [turnFalse turnRange neg]; % negative matrix
    negative = [negative; 0 1024 -1];     % add the maximum possible value
    
    % zero value
    forwdRange = [0:10:(darker+200)]';          % move forward range
    matrixLength = length(forwdRange);          % store the length
    forwdZeros = zeros(matrixLength, 1);        % negative forward range
    forwdOnes = ones(matrixLength, 1);          % positive forward range
    zero1 = [forwdZeros forwdRange forwdZeros]; % negative matrix
    zero2 = [forwdOnes forwdRange forwdZeros];  % negative matrix
    zero = [zero1; zero2];                      % combine the matrices
    
    trainingData = [positive; negative; zero];  % concatenate training data
end
