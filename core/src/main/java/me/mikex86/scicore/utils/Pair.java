package me.mikex86.scicore.utils;

public class Pair<F, S> {

    private final F first;
    private final S second;

    public Pair(F first, S second) {
        this.first = first;
        this.second = second;
    }

    public static <F, S> Pair<F, S> of(F f, S s) {
        return new Pair<>(f, s);
    }

    public F getFirst() {
        return first;
    }

    public S getSecond() {
        return second;
    }

}
