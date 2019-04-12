#include <ctime>
#include <cmath>
#include <random>
#include <thread>

#include <SFML/Graphics.hpp>

#include "Basket.hpp"
#include "Food.hpp"
#include "Game.hpp"

void Game::reset(sf::RenderWindow &window) {
    std::mt19937_64 re(std::time(0));
    std::uniform_real_distribution<double> velo(-MAX_SPEED_COMPONENT, MAX_SPEED_COMPONENT);
    std::uniform_real_distribution<double> x(0, window.getSize().x);
    std::uniform_real_distribution<double> y(0, window.getSize().y);

    basket->setPos(*(new sf::Vector2f(x(re), y(re))));
    food->reset(*(new sf::Vector2f(x(re), y(re))), *(new sf::Vector2f(velo(re), velo(re))));
}

Game::Game(sf::RenderWindow &window) {
    fitnesses.resize(CREATURES);
    food = new Food();
    basket = new Basket();
    reset(window);
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
    // kill bottom 80%, replace with mutated versions of top 20%
}

void Game::generation(sf::RenderWindow &window, unsigned long long int tickFreq, bool &stop) {
    for (int i = 0; i < CREATURES; ++i) {
        reset(window);
        for (int j = 0; j < TICKS_PER_GENERATION; ++j) {
            tick(window);
            if (stop) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(tickFreq));
        }
        fitnesses[i] = fitness();
        // load next brain:
    }
    mutate();
}