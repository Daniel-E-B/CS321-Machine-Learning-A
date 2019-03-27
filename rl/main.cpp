#include <SFML/Graphics.hpp>
#include <iostream>

#include "Basket.hpp"
#include "Player.hpp"

int main() {
    const int WIDTH = 1280, HEIGHT = 720;

    sf::ContextSettings ctx;
    ctx.antialiasingLevel = 32;
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "window", sf::Style::Default, ctx);
    window.setFramerateLimit(60);
    window.setVerticalSyncEnabled(true);

    Player p(*(new sf::Vector2f(WIDTH / 2, HEIGHT / 2))); // there has to be a better way
    Basket b(*new sf::Vector2f(WIDTH/2, HEIGHT));

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }
        window.clear(sf::Color::Black);
        p.draw(window);
        b.draw(window);
        window.display();
    }
    return 0;
}