%Jay Maini 101037537
%CMPA
close all
set(0, 'DefaultFigureWindowStyle', 'docked')

Is = 0.01e-12
Ib = 0.1e-12
Vb = 1.3
Gp = 0.1


%Step 1 - Data generation
V = linspace(-1.95, 0.7, 200);
I = Is.*(exp(((1.2.*V)/0.025))-1) + Gp.*V - Ib.*(exp((-1.2/0.025).*(V+Vb))-1); 
I_variation = (0.4*rand+0.8)*I;

p4 = polyfit(V,I,4);
exp4 = polyval(p4,V);
p8 = polyfit(V,I,8);
exp8 = polyval(p8,V);

subplot(3,2,1)
plot(V,I,V,I_variation,V,exp4,V,exp8)
legend('data','varied','Poly 4', 'Poly 8')
xlabel('V')
ylabel('I')
subplot(3,2,2)
semilogy(V,I,V,I_variation,V,exp4,V,exp8)
xlabel('V')
ylabel('I')
legend('data','varied','Poly 4', 'Poly 8')

%Higher order polyfits work better!

%Using fit (three)
f1 = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
f2 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
f3 = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');

ff1 = fit(V.',I.',f1);
ff2 = fit(V.',I.',f2);
ff3 = fit(V.',I.',f3);

If1 = ff1(V);
If2 = ff2(V);
If3 = ff3(V)

subplot(3,2,3)
plot(V,I,V,If1,V,If2,V,If3)
legend('data','fit1','fit2','fit3')
xlabel('V')
ylabel('I')
subplot(3,2,4)
semilogy(V,I,V,If1,V,If2,V,If3)
legend('data','fit1','fit2','fit3')
xlabel('V')
ylabel('I')


%Neural Net Model
inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
view(net)
Inn = outputs;

subplot(3,2,5)
plot(V,I,V,Inn)
legend('data','neural fit')
xlabel('V')
ylabel('I')
subplot(3,2,6)
semilogy(V,I,V,Inn)
legend('data','neural fit')
xlabel('V')
ylabel('I')


