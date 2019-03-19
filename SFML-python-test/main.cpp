#include <chrono>
#include <iostream>
#include <SFML/Graphics.hpp>
#include <boost/python.hpp>

int main() {
    sf::ContextSettings settings;
    settings.antialiasingLevel = 16;
    sf::RenderWindow window(sf::VideoMode(480, 480), "test", sf::Style::Default, settings);
    window.setFramerateLimit(60);
    window.setVerticalSyncEnabled(true);
    sf::CircleShape c(50, 1000);
    while (window.isOpen()) {
        long time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }
        c.setPosition((int(c.getPosition().x) + 1) % window.getSize().x,
                      (int(c.getPosition().y) + 1) % window.getSize().y);
        c.setFillColor(sf::Color(c.getFillColor().r + 2, c.getFillColor().g + 1, c.getFillColor().b + 0.5));
        window.clear(sf::Color::Black);
        window.draw(c);
        window.display();
        std::cout << (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() - time) << std::endl;
    }
    return 0;
}