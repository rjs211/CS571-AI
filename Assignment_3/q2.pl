xor_opn(0, 0, 0). xor_opn(0, 1, 1). xor_opn(1, 0, 1). xor_opn(1, 1, 0).
or_opn(0, 0, 0). or_opn(0, 1, 1). or_opn(1, 0, 1). or_opn(1, 1, 1).
and_opn(0, 0, 0). and_opn(0, 1, 0). and_opn(1, 0, 0). and_opn(1, 1, 1).


% type(xor11, xor).
% type(xor12, xor).

% type(xor21, xor).
% type(xor22, xor).

% type(and11, and).
% type(and12, and).

% type(and21, and).
% type(and22, and).

% type(or11, or).
% type(or21, or).

type(xor11, xor).
type(xor12, xor).

type(xor21, xor).
type(xor22, xor).

type(and11, and).
type(and12, and).

type(and21, and).
type(and22, and).

type(or11, or).
type(or21, or).


terminal(xor11_in1).
terminal(xor11_in2).
terminal(xor11_out).

terminal(xor12_in1).
terminal(xor12_in2).
terminal(xor12_out).

terminal(xor21_in1).
terminal(xor21_in2).
terminal(xor21_out).

terminal(xor22_in1).
terminal(xor22_in2).
terminal(xor22_out).

terminal(and11_in1).
terminal(and11_in2).
terminal(and11_out).

terminal(and12_in1).
terminal(and12_in2).
terminal(and12_out).

terminal(and21_in1).
terminal(and21_in2).
terminal(and21_out).

terminal(and22_in1).
terminal(and22_in2).
terminal(and22_out).

terminal(or11_in1).
terminal(or11_in2).
terminal(or11_out).

terminal(or21_in1).
terminal(or21_in2).
terminal(or21_out).


in(1, xor11, xor11_in1).
in(2, xor11, xor11_in2).

in(1, xor12, xor12_in1).
in(2, xor12, xor12_in2).

in(1, xor21, xor21_in1).
in(2, xor21, xor21_in2).

in(1, xor22, xor22_in1).
in(2, xor22, xor22_in2).

in(1, and11, and11_in1).
in(2, and11, and11_in2).

in(1, and12, and12_in1).
in(2, and12, and12_in2).

in(1, and21, and21_in1).
in(2, and21, and21_in2).

in(1, and22, and22_in1).
in(2, and22, and22_in2).

in(1, or11, or11_in1).
in(2, or11, or11_in2).

in(1, or21, or21_in1).
in(2, or21, or21_in2).


out(xor11, xor11_out).
out(xor12, xor12_out).
out(xor21, xor21_out).
out(xor22, xor22_out).
out(and11, and11_out).
out(and12, and12_out).
out(and21, and21_out).
out(and22, and22_out).
out(or11, or11_out).
out(or21, or21_out).


connected(x1, xor11_in1).
connected(x2, xor11_in2).

connected(c1, xor12_in2).

connected(xor11_out, xor12_in1).
connected(xor12_out, z1).

connected(x1, and12_in1).
connected(x2, and12_in2).

% connected(xor11_out, and11_in1).
connected(c1, and11_in2).

connected(and11_out, or11_in1).
connected(and12_out, or11_in2).

connected(or11_out, c2).



connected(y1, xor21_in1).
connected(y2, xor21_in2).

connected(c2, xor22_in2).

connected(xor21_out, xor22_in1).
connected(xor22_out, z2).

connected(y1, and22_in1).
connected(y2, and22_in2).

connected(xor21_out, and21_in1).
connected(c2, and21_in2).

connected(and21_out, or21_in1).
connected(and22_out, or21_in2).

connected(or21_out, c3).


% signal(and11_in1, 1).
% signal(and11_in2, 0).

% signal(or11_in1, 1).
% signal(or11_in2, 0).

% signal(xor11_in1, 1).
% signal(xor11_in2, 0).

signal(x1, 1).
signal(x2, 1).
signal(y1, 1).
signal(y2, 1).
signal(c1, 1).

% out(X, Y), in(1, X, Z1), in(2, X, Z2) , signal(Y) is xor_opn(signal(Z1), signal(Z2)) :- type(X, xor).
signal(Terminal, OutVar) :- out(X, Terminal), type(X, xor), in(1, X, In1), in(2, X, In2), signal(In1, In1v), signal(In2, In2v), xor_opn(In1v, In2v, OutVar).

signal(Terminal, OutVar) :- out(X, Terminal), type(X, or), in(1, X, In1), in(2, X, In2), signal(In1, In1v), signal(In2, In2v), or_opn(In1v, In2v, OutVar).

signal(Terminal, OutVar) :- out(X, Terminal), type(X, and), in(1, X, In1), in(2, X, In2), signal(In1, In1v), signal(In2, In2v), and_opn(In1v, In2v, OutVar).

% signals_equal(X, Y) :- connected(X, Y)
% connected(Y, X) :- connected(X, Y)

% signals_equal(X, Y) :- signal(X, X_sig), signal(Y, Y_sig), X_sig == Y_sig.

signal(X, Y) :- connected(Z, X), signal(Z, Y).


fault_(X) :- connected(X, Z), !, fail.
fault_(X) :- connected(Z, X), !, fail. 
fault_(X) :- terminal(X).
fault(X) :- terminal(X), fault_(X).
