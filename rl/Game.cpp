#include <chrono>
#include <cmath>
#include <random>
#include <thread>

#include <SFML/Graphics.hpp>

#include "Basket.hpp"
#include "Food.hpp"
#include "Game.hpp"

Game::Game(sf::RenderWindow &window) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-MAX_SPEED_COMPONENT, MAX_SPEED_COMPONENT);

    food = new Food(*(new sf::Vector2f(window.getSize().x / 2, window.getSize().y / 2)), *(new sf::Vector2f(dist(mt), (dist(mt)))));
    basket = new Basket(*(new sf::Vector2f(window.getSize().x / 2, window.getSize().y / 2)));

    fitnesses.resize(CREATURES);
}

void Game::draw(sf::RenderWindow &window) {
    food->draw(window);
    basket->draw(window);
}

void Game::tick(sf::RenderWindow &window) {
    food->move(window);
    basket->move(*(new sf::Vector2f(food->getPos().x / window.getSize().x, food->getPos().y / window.getSize().y)));
}

double Game::fitness() {
    return (sqrt(pow(food->getPos().x - basket->getPos().x, 2) + pow(food->getPos().y - basket->getPos().y, 2)));
}

void Game::mutate() {
}

void Game::generation(sf::RenderWindow &window, unsigned long long int tickFreq, bool &stop) {
    for (int i = 0; i < CREATURES; ++i) {
        // reset(); TODO: reset positions, load next brain
        for (int j = 0; j < TICKS_PER_GENERATION; ++j) {
            tick(window);
            if (stop) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(tickFreq));
        }
        // compute fitness:
        fitnesses[i] = fitness();
    }
}