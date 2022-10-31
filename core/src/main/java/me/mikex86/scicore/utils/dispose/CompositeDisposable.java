package me.mikex86.scicore.utils.dispose;

import org.jetbrains.annotations.NotNull;

/**
 * A {@link CompositeDisposable} is a collection of {@link IDisposable}s that must be disposed of together.
 */
public class CompositeDisposable implements IDisposable {

    @NotNull
    private final IDisposable[] disposables;

    private CompositeDisposable(IDisposable @NotNull [] disposables) {
        this.disposables = disposables;
    }

    @NotNull
    public static CompositeDisposable of(IDisposable @NotNull ... disposables) {
        return new CompositeDisposable(disposables);
    }

    @Override
    public void dispose() {
        for (IDisposable disposable : disposables) {
            disposable.dispose();
        }
    }
}
