package me.mikex86.scicore.profiling;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

public class Profiler {


    private static class Scope {

        @Nullable
        private final Scope parent;

        private final String name;
        private long totalTimeSpent;

        private final Map<String, Scope> children = new HashMap<>();

        public Scope(@Nullable Scope parent, String name) {
            this.parent = parent;
            this.name = name;
        }

        public Scope getChild(String name) {
            return children.computeIfAbsent(name, x -> new Scope(this, name));
        }

        public void addTime(long time) {
            totalTimeSpent += time;
        }

        @Nullable
        public Scope getParent() {
            return parent;
        }

        public long getTotalTimeSpent() {
            return totalTimeSpent;
        }

        public String getName() {
            return name;
        }
    }

    private static final Scope rootScope = new Scope(null, "root");

    private static Scope currentScope = rootScope;

    private static final Stack<Long> startTimes = new Stack<>();

    public static void startScope(@NotNull String name) {
        currentScope = currentScope.getChild(name);
        startTimes.push(System.nanoTime());
    }

    public static void endScope(@NotNull String name) {
        if (!currentScope.getName().equals(name)) {
            throw new IllegalArgumentException("endScope() called with a different name than startScope().");
        }
        long endTime = System.nanoTime();
        long startTime = startTimes.pop();
        currentScope.addTime(endTime - startTime);
        currentScope = currentScope.getParent();
    }

    public static void printStats(@NotNull Scope scope, int indent) {
        for (int i = 0; i < indent; i++) {
            System.out.print(" ");
        }
        long unProfiledTime = scope.getTotalTimeSpent();
        for (Scope child : scope.children.values()) {
            unProfiledTime -= child.getTotalTimeSpent();
        }

        double profiledSeconds = scope.getTotalTimeSpent() / 1e9;
        double unProfiledPercent = unProfiledTime / (double) scope.getTotalTimeSpent() * 100;
        System.out.println(">" + scope.name + ": " + String.format("%.3f", profiledSeconds) + "s" + (!scope.children.isEmpty() ? " (" + String.format("%.3f", unProfiledPercent) + "% unprofiled)" : ""));
        for (Scope child : scope.children.values()) {
            printStats(child, indent + 2);
        }
    }

    public static void printStats() {
        printStats(rootScope, 0);
    }
}
