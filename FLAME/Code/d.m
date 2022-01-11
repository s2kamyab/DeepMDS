function dist = d(x , y , distType)
global down up population;
switch distType
    case 1%euclidian
        dist = sqrt(sum((x - y).^ 2 , 2));
    case 2%fuzzy rough
        id = 1:size(x , 2);
%         dist = 1-lfuzzyrough_p([id' , x' , y'])'
        dist = 1-lfuzzyrough([id' , x' , y'])';
    case 3%fuzzy hamming
        xMapped = (x - repmat(down,size(x , 1) , 1)) ./ repmat(up - down , size(x , 1) , 1);
        yMapped = (y - repmat(down,size(x , 1) , 1)) ./ repmat(up - down , size(x , 1) , 1);
        dist = sum(abs(xMapped - yMapped) , 2);
    case 4% normalized fuzzy hamming
        xMapped = (x - repmat(down,size(x , 1) , 1)) ./ repmat(up - down , size(x , 1) , 1);
        yMapped = (y - repmat(down,size(x , 1) , 1)) ./ repmat(up - down , size(x , 1) , 1);
        dist = (1/size(x , 2)) .* sum(abs(xMapped - yMapped) , 2);
    case 5%Hausdorff metric
        xMapped = (x - repmat(down,size(x , 1) , 1)) ./ repmat(up - down , size(x , 1) , 1);
        yMapped = (y - repmat(down,size(x , 1) , 1)) ./ repmat(up - down , size(x , 1) , 1);
        dist = max(abs(xMapped - yMapped)')';
    case 6%analogous of Gregson's model in 192-202 in proximity/sim
        xMapped = (x - repmat(down,size(x , 1) , 1)) ./ repmat(up - down , size(x , 1) , 1);
        yMapped = (y - repmat(down,size(x , 1) , 1)) ./ repmat(up - down , size(x , 1) , 1);
        dist = 1 - sum(min(xMapped , yMapped) , 2)./sum(max(xMapped , yMapped) , 2);
    case 7% eq 24 in fuzzy-rough
        temp = abs(x - y)./abs(repmat(up - down , size(x , 1) , 1));
        dist = max(temp')';
    case 8% eq 25 in fuzzy rough
        sigma = sqrt(var(population));
        temp = -exp(- ((x - y).^ 2) ./ (2*(repmat(sigma , size(x , 1) , 1).^2)));
        dist = -max(temp')';
    case 9 % eq 26 in fuzzy rough
        sigma = sqrt(var(population));
        sigma = repmat(sigma , size(x , 1) , 1);
        temp1 = (y - (x - sigma))./ (x - (x - sigma));
        temp2 = ((x + sigma)-y) ./ (x + sigma - x);
        temp3 = 1 - max(min(temp1 , temp2) , 0);
        dist = max(temp3')';
    case 10 % The analogous of Restle's model,page 5 from 192-202
        xMapped = (x - repmat(down,size(x , 1) , 1)) ./ repmat(up - down , size(x , 1) , 1);
        yMapped = (y - repmat(down,size(x , 1) , 1)) ./ repmat(up - down , size(x , 1) , 1);
        temp1 = min(xMapped , 1 - yMapped);
        temp2 = min(yMapped , 1-xMapped);
        temp3 = max(temp1 , temp2);
        dist = sum(temp3 , 2);
    case 11 % hamming
        dist = sum(xor(X , y) , 2);
    otherwise
        error('The distance you choose doesn''t exist');
end