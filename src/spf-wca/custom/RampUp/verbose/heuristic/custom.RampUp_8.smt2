(declare-const in6 Int)
(declare-const in5 Int)
(declare-const in7 Int)
(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in4 Int)
(declare-const in3 Int)

(assert (and (and (and (and (and (and (and  ( <  in0 0)  ( <  in1 0))  ( <  in2 0))  ( <  in3 0))  ( <  in4 0))  ( <  in5 0))  ( <  in6 0))  ( <  in7 0)))

(check-sat)
(get-model)