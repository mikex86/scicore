package me.mikex86.scicore;

import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import me.tongfei.progressbar.ProgressBarStyle;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class MNISTTest {

    @NotNull
    private static final Path MNIST_DIR = Path.of("mnist");

    private static void downloadMnist() throws IOException, InterruptedException {
        if (MNIST_DIR.toFile().exists()) {
            System.out.println("MNIST already downloaded");
            return;
        }
        HttpClient client = HttpClient.newHttpClient();
        List<String> urls = List.of(
                "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
        );
        Files.createDirectories(MNIST_DIR);

        for (String url : ProgressBar.wrap(urls, new ProgressBarBuilder()
                .setStyle(ProgressBarStyle.ASCII)
                .setTaskName("Downloading MNIST"))) {
            String filename = url.substring(url.lastIndexOf('/') + 1);
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .GET()
                    .build();
            client.send(request, HttpResponse.BodyHandlers.ofFile(MNIST_DIR.resolve(filename)));
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        downloadMnist();
    }

}
