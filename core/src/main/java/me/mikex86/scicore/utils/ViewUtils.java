package me.mikex86.scicore.utils;

import me.mikex86.scicore.tensor.ITensor;
import me.mikex86.scicore.tensor.View;
import org.jetbrains.annotations.NotNull;

public class ViewUtils {
    public static long getTotalOffset(@NotNull View view) {
        long totalOffset = 0;
        ITensor viewed = view;
        while (viewed instanceof View) {
            totalOffset += view.getOffset();
            viewed = view.getViewed();
        }
        return totalOffset;
    }

    @NotNull
    public static ITensor getViewed(@NotNull View view) {
        ITensor viewed = view;
        while (viewed instanceof View viewedView) {
            viewed = viewedView.getViewed();
        }
        return viewed;
    }
}
