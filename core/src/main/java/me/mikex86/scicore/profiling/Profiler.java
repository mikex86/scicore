package me.mikex86.scicore.profiling;

import org.jetbrains.annotations.NotNull;

import java.util.Stack;

public class Profiler {

    public static final boolean USE_PROFILER = false;

    @NotNull
    private final Stack<String> sections = new Stack<>();


    public void push(@NotNull String sectionName) {
        this.sections.push(sectionName);
    }

    public void pop() {
        this.sections.pop();
    }

}
