(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in3 Int)

(assert (and (and (and (and (and  ( <=  in1 in0)  ( <  in1 in0))  ( <=  in2 in1))  ( <  in2 in1))  ( <=  in3 in2))  ( <  in3 in2)))

(check-sat)
(get-model)