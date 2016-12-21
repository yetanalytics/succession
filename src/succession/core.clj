(ns succession.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.stats :as ms]))

(defn log-likelihood* [{:keys [vt ft at pt t z h]} y]
  (let [v  (m/sub y (m/mmul z at))
        zt (m/transpose z)
        f  (inc (first (m/mmul z pt zt)))
        kt (-> (m/mmul t pt zt)
               (m/add h)
               (m/div f))
        at (m/add (m/mmul t at)
                  (m/mmul kt v))
        lt (m/sub t (m/mmul kt z))
        jt (m/sub h kt)
        pt (m/add (m/mmul t pt (m/transpose lt))
                  (m/mmul h (m/transpose jt)))]
    {:vt (cons v vt)
     :ft (cons f ft)
     :at at :pt pt :t t :h h :z z}))

(defn log-likelihood [coefs p q ys]
  (let [m     (max p q)
        phi   (take m (concat (take p coefs) (repeat 0)))
        theta (take m (concat (drop p coefs) (repeat 0)))
        init {:at (m/new-matrix m 1)
              :pt (m/identity-matrix m)
              :t  (m/join-along 1
                   (m/matrix phi)
                   (m/join-along 0
                    (m/identity-matrix (dec m))
                    (m/new-matrix 1 (dec m))))
              :h (m/add phi theta)
              :z (first (m/identity-matrix m))}
        ll    (reduce log-likelihood* init ys)
        sigma (-> (m/square  (:vt ll))
                  (m/div (:ft ll))
                  (ms/mean))
        n (count ys)]
    (* -0.5 (+ n
               (* n (m/log (* 2 Math/PI)))
               (* n (m/log sigma))
               (ms/sum (m/log (:ft ll)))))))

(defn aic [coefs p q ys]
  (+ (* -2 (log-likelihood coefs p q ys))
     (* 2 (+ p q 1))))
