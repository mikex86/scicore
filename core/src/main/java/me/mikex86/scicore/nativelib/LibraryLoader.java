package me.mikex86.scicore.nativelib;

import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

public class LibraryLoader {

    public static void loadLibrary(@NotNull String libraryName) {
        // Extract $libraryName.dll / lib$libraryName.so / lib$libraryName.dylib from the classpath and load it in JVM
        ClassLoader classLoader = LibraryLoader.class.getClassLoader();
        String fileName = System.mapLibraryName(libraryName);
        try (InputStream inputStream = classLoader.getResourceAsStream("natives/" + fileName)) {
            if (inputStream == null) {
                throw new RuntimeException("Library not found in classpath: " + libraryName);
            }
            Path tempFile = Files.createTempFile(fileName, null);
            tempFile.toFile().deleteOnExit();
            Files.copy(inputStream, tempFile, StandardCopyOption.REPLACE_EXISTING);
            System.load(tempFile.toAbsolutePath().toString());
        } catch (IOException e) {
            throw new RuntimeException("Error while loading library: " + fileName, e);
        }
    }
}
