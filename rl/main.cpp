#include <iostream>
#include <thread>
#include <vector>

#include <SFML/Graphics.hpp>

#include "Game.hpp"
#include "tiny-dnn/tiny_dnn/tiny_dnn.h"

void doGenerations(sf::RenderWindow &window, Game &g, int numGenerations, bool &stop) {
    for(int i  = 0; i < numGenerations; ++i){
        g.generation(window, 10, stop); // milliseconds of delay between ticks (make it variable? / user selectable?)
    }
}

int main() {
    // network library testing:
    ////////////////////////////////////////////////////////////////////////////////
    tiny_dnn::network<tiny_dnn::sequential> net;
    tiny_dnn::adagrad opt;
    net << tiny_dnn::layers::fc(2, 3) << tiny_dnn::activation::leaky_relu()
        << tiny_dnn::layers::fc(3, 1) << tiny_dnn::activation::sigmoid();

    std::vector<tiny_dnn::vec_t> x_train{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<tiny_dnn::vec_t> y_train{{0}, {1}, {1}, {0}};
    size_t batch_size = 2;
    size_t epochs = 300;

    net.fit<tiny_dnn::mse>(opt, x_train, y_train, batch_size, epochs);
    tiny_dnn::vec_t result = net.predict({1, 0});
    // std::cout << result[0] << std::endl;
    result = net.predict({1, 1});
    // std::cout << result[0] << std::endl;
    net.save("net.json", tiny_dnn::content_type::weights, tiny_dnn::file_format::json);
    // net.load("net", tiny_dnn::content_type::weights, tiny_dnn::file_format::json);
    ////////////////////////////////////////////////////////////////////////////////

    const int WIDTH = 1280, HEIGHT = 720;

    sf::ContextSettings ctx;
    ctx.antialiasingLevel = 32;
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "window", sf::Style::Default, ctx);
    window.setFramerateLimit(60);
    window.setVerticalSyncEnabled(true);

    Game g(window);
    bool stopGens = false;
    std::thread gameThread(doGenerations, std::ref(window), std::ref(g), 10, std::ref(stopGens));
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                stopGens = true;
                gameThread.join();
                window.close();
            }
        }
        window.clear(sf::Color::Black);
        g.draw(window);
        window.display();
        // TODO: call generation() in a thread, so that the window will not freeze. might need to mutex calls to window.*
    }
    return 0;
}