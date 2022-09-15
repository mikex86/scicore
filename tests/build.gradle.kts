plugins {
    id("java")
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven(url = "https://packages.jetbrains.team/maven/p/skija/maven")
}

dependencies {
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.9.0")
    testImplementation("org.junit.jupiter:junit-jupiter-params:5.9.0")
    testImplementation("org.junit.platform:junit-platform-suite-api:1.9.0")
    testImplementation("org.junit.jupiter:junit-jupiter-engine")

    testImplementation("me.tongfei:progressbar:0.9.4")

    implementation("org.jetbrains:annotations:23.0.0")
    implementation(project(":matplotlib"))
    implementation(project(":core"))
    implementation(project(":genericcpu-backend"))
    implementation(project(":cuda-backend"))

}

tasks.getByName<Test>("test") {
    useJUnitPlatform()
}