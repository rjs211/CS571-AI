member(a).
member(b).
member(c).

% Every member in the club is either a mc or a skier or both
member(X) :- notmc(X), notskier(X), !, fail.
member(X).

% A likes rain and snow
like(a, rain).
like(a, snow).

% A likes whatever B dislikes and dislikes whatever B like
like(a, X) :- dislike(b, X).
like(b, X) :- dislike(a, X).

% No mountain climber likes the rain
% If mc, not like rain
% \+ like(X, rain) :- mc(X)
% like(X, rain) :- mc(X), !, fail.
% like(X, rain).

mc(X) :- like(X, rain), !, fail.
mc(X) :- member(X), notskier(X).
% mc(X).  % not fully correct

% Every skier likes snow
% likes(X, snow) :- skier(X).
notskier(X) :- dislike(X, snow).

dislike(P,Q) :- like(P,Q), !, fail.
dislike(P,Q).

notskier(X) :- skier(X), !, fail.
notskier(X).
notmc(X) :- mc(X), !, fail.
notmc(X).

query(X) :- member(X), mc(X), notskier(X), !.
